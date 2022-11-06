# vision_trainer_2d
# 参考  

vit_pytorchでのtrain
https://farml1.com/vit_2/

モデルの保存など  
https://tzmi.hatenablog.com/entry/2020/03/05/222813

モデルの推論
https://pystyle.info/pytorch-how-to-use-pretrained-model/#outline__3_4

# ファイル構成
| ファイル名 | 内容 |
| ---- | ---- |
| main_dl_and_predict.py | メインファイル。各関数を読み出して実行、予測、GCSへの保存などを行う。|
| get_imgs_from_tw.py | TWのタイムラインから画像を取得し、保存する |
| predict.py | 指定したディレクトリに対して予測を行う | 
| gcp_controller.py | gcp上の操作を行う。画像の保存など |
| utils.py | 細かい関数やクラスなど |
| model_environment.py | モデル内の設定。モデルをいじるときはここをいじる。 | 
| train_by_vitpytorch.py | 学習を行う |