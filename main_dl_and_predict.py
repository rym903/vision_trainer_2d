from get_imgs_from_tw import init_auth, download_from_timeline
from predict import predict
from gcp_controller import to_storage

if __name__ == "__main__":
    driver, api = init_auth()
    dl_path = download_from_timeline(driver, api)
    df_true, df_false = predict(dl_path)
    to_storage(df_true, df_false)
