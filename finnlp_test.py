from finnlp.data_sources.news.finnhub_date_range import Finnhub_Date_Range

start_date = "2023-01-01"
end_date = "2023-01-03"
config = {
    "use_proxy": "us_free",  # use proxies to prvent ip blocking
    "max_retry": 5,
    "proxy_pages": 5,
    "token": "YOUR_FINNHUB_TOKEN",  # Available at https://finnhub.io/dashboard
}

news_downloader = Finnhub_Date_Range(config)  # init
news_downloader.download_date_range_stock(start_date, end_date)  # Download headers
news_downloader.gather_content()  # Download contents
df = news_downloader.dataframe
selected_columns = ["headline", "content"]
df[selected_columns].head(10)
