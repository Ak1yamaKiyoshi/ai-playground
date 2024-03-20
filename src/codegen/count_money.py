import os
import re 
all_logs = ""
for file in os.listdir("logs"):
    with open(f"logs/{file}", "r") as f:
        all_logs += "\n" + f.read()

all_logs = [log for log in all_logs.split("\n") if log.strip() != ""]
price_index = all_logs[0].find("price $")
prices = []
regexp = re.compile(r"[^\d.]+")
for line in all_logs:
    prices.append(line[price_index+7:price_index+13])
prices = [regexp.sub("", price) for price in prices]
prices = [float(price) for price in prices]
print(f"Total from logs: ${round(sum(prices), 3)}")
