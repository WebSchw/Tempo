import requests


csv_urls = [
    "https://raw.githubusercontent.com/recski/brise-plandok/main/brise_plandok/baselines/input/test_data.csv",
    "https://raw.githubusercontent.com/recski/brise-plandok/main/brise_plandok/baselines/input/train_data.csv",
    "https://raw.githubusercontent.com/recski/brise-plandok/main/brise_plandok/baselines/input/valid_data.csv"
]
for url in csv_urls:
    r = requests.get(url, allow_redirects=True)
    set = url.split("/")[-1].split("_")[0]
    open(f'{set}_data.csv', 'wb').write(r.content)



