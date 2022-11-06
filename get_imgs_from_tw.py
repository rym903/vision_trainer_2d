import datetime
import os
import re
import tweepy
import pandas as pd

# from bs4 import BeautifulSoup
from dotenv import load_dotenv
from selenium import webdriver

from utils import download_file, get_img_urls


def init_auth():
    """seleniumの初期化と、twitter APIのauth

    Returns:
        driver: selenium driver
        api: twitter api (認証済み)
    """
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome("chromedriver", options=options)

    # 認証情報の読み込み
    load_dotenv()
    consumer_key = os.getenv("API_KEY")
    consumer_secret = os.getenv("API_SECRET_KEY")
    access_key = os.getenv("ACCESS_TOKEN")
    access_secret = os.getenv("ACCESS_TOKEN_SECRET")

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
    return driver, api


def download_from_timeline(driver, api):
    """twitterのタイムラインから画像を抜き出し、保存する

    Args:
        driver (_type_): _description_
        api (_type_): _description_
    Returns:
        img_dir: 保存先のパス
    """
    # ホーム画面のタイムラインからTW取得
    tweet_data = []
    account_name = "creantics"
    for tweet in api.home_timeline(
        screen_name=account_name, count=10, tweet_mode="extended"
    ):
        tweet_data.append(
            [
                tweet.id,
                tweet.created_at,
                tweet.full_text.replace("\n", ""),
                tweet.user.screen_name,
                # tweet.favorite_count,
                # tweet.retweet_count,
            ]
        )

    df_tw = pd.DataFrame(
        tweet_data,
        columns=[
            "id",
            "datetime",
            "text",
            "id_name",
            "favorite_count",
            "retweet_count",
        ],
    )

    # urlを取り出す
    links = []
    ids = []
    for index, record in df_tw.iterrows():
        match = re.search(r"(https://t.co/.{10})", record["text"])
        if not match:
            continue
        links.append(match.groups()[0])
        ids.append(record["id_name"])

    # 保存するフォルダ作成
    today = datetime.datetime.today()
    dt = today.strftime("%Y%m%d")
    img_dir = f"./img/img_{dt}"
    try:
        os.makedirs(img_dir)
    except FileExistsError as e:
        print(f"{img_dir}はすでに存在します")

    # 保存
    for link, id in zip(links, ids):
        img_urls = get_img_urls(link, driver)
        for cnt, img_url in enumerate(img_urls):
            dttime = today.strftime("%Y%m%d-%H%M%S")
            dst_path = f"{img_dir}/{id}_{cnt:0=4}_{dttime}.jpg"
            download_file(img_url, dst_path)

    return img_dir
