import os
import pandas as pd
import datetime

from glob import glob
from google.cloud import storage as gcs
from google.oauth2 import service_account
from io import BytesIO

project_id = "strategic-altar-207405"
bucket_id = "tw_predictor"


class GCSWrapper:
    def __init__(self, project_id, bucket_id, from_local=False):
        """GCSのラッパークラス
        Arguments:
            project_id {str} -- GoogleCloudPlatform Project ID
            bucket_id {str} -- GoogleCloudStorage Bucket ID
            from_local (bool) -- ローカル実行か否か
        """
        self._project_id = project_id
        self._bucket_id = bucket_id
        if from_local:
            credential = service_account.Credentials.from_service_account_file(
                "/Users/ryomanakagawa/prac/transformers_tutorial/secrets/gcp_read_and_wrote_from_python.json"
            )
            self._client = gcs.Client(project_id, credentials=credential)
        else:
            self._client = gcs.Client(project_id)
        self._bucket = self._client.get_bucket(self._bucket_id)

    def exists(self, dir_name) -> bool:
        """dirの存在確認"""
        if not dir_name:
            try:
                _ = self._bucket
                return True
            except ValueError:
                return False
        return bool(self._bucket.get_blob(dir_name))

    def show_bucket_names(self):
        """バケット名の一覧を表示"""
        [print(bucket.name) for bucket in self._client.list_buckets()]

    def show_file_names(self):
        """バケット内のファイル一覧を表示"""
        [print(file.name) for file in self._client.list_blobs(self._bucket)]

    def upload_file(self, local_path, gcs_path):
        """GCSにローカルファイルをアップロード

        Arguments:
            local_path {str} -- local file path
            gcs_path {str} -- gcs file path
        """
        blob = self._bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)

    def upload_file_as_dataframe(self, df, gcs_path, flg_index=False, flg_header=True):
        """GCSにpd.DataFrameをCSVとしてアップロード

        Arguments:
            df {pd.DataFrame} -- DataFrame for upload
            gcs_path {str} -- gcs file path

        Keyword Arguments:
            flg_index {bool} -- DataFrame index flg (default: {False})
            flg_header {bool} -- DataFrame header flg (default: {True})
        """
        blob = self._bucket.blob(gcs_path)
        blob.upload_from_string(df.to_csv(index=flg_index, header=flg_header, sep=","))

    def download_file(self, local_path, gcs_path):
        """GCSのファイルをファイルとしてダウンロード

        Arguments:
            local_path {str} -- local file path
            gcs_path {str} -- gcs file path
        """
        blob = self._bucket.blob(gcs_path)
        blob.download_to_filename(local_path)

    def download_file_as_dataframe(self, gcs_csv_path):
        """GCSのファイルをpd.DataFrameとしてダウンロード

        Arguments:
            gcs_csv_path {str} -- gcs file path (only csv file)

        Returns:
            [pd.DataFrame] -- csv data as pd.DataFrame
        """
        blob = self._bucket.blob(gcs_csv_path)
        content = blob.download_as_string()
        df = pd.read_csv(BytesIO(content))
        return df

    def upload_all_files(self, dest_dir, local_dir):
        """ローカルから指定したdirのフォルダを、dist_dirにアップロードする

        Args:
            bucket_namse (_type_): _description_
            local_files (_type_): _description_
        """
        # ファイル名の取得
        file_paths = glob(local_dir)

        # 保存
        for local_file_path in file_paths:
            true_false = os.path.basename(os.path.dirname(local_file_path))
            file_name = os.path.basename(local_file_path)
            self.upload_file(
                local_file_path, os.path.join(dest_dir, true_false, file_name)
            )


def to_storage(df_true, df_false):
    """予測結果をアップロード

    Args:
        df_true (_type_): _description_
        df_false (_type_): _description_
    """
    gcs_driver = GCSWrapper(project_id, bucket_id)
    dest_dir = datetime.datetime.today().strftime("%Y%m%d")

    for local_file_path in df_true["path"].iteritems():
        gcs_driver.upload_file(
            local_file_path, os.path.join(dest_dir, "true", local_file_path)
        )

    for local_file_path in df_false["path"].iteritems():
        gcs_driver.upload_file(
            local_file_path, os.path.join(dest_dir, "false", local_file_path)
        )
