import requests
import csv

API_URL="https://archive-api.open-meteo.com/v1/archive?latitude=52.52&longitude=13.41&start_date=2025-11-15&end_date=2025-11-29&hourly=temperature_2m"
response=requests.get(API_URL)
data=response.json()
# print(data)
times = data["hourly"]["time"]
temps = data["hourly"]["temperature_2m"]
with open("weather_api.csv","w",newline="") as wcsv:
    csv_writer=csv.writer(wcsv)
    csv_writer.writerow(["time", "temperature"])
    for t, temp in zip(times, temps):
        csv_writer.writerow([t, temp])