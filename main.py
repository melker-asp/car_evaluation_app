import requests
from bs4 import BeautifulSoup
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import csv

def get_make():
    while True:
        make = input("Ange bilmärke: ").strip()
        if make:
            return make
        print("Vänligen ange ett giltigt bilmärke.")

def get_model():
    while True:
        model = input("Ange modell: ").strip()
        if model:
            return model
        print("Vänligen ange en giltig modell")

MAKE = get_make()
MODEL = get_model()

def get_year(prompt, min_year=1920, max_year=2025):
    while True:
        try:
            year = int(input(prompt).strip())
            if min_year <= year <= max_year:
                return year
            print(f"Ange ett år mellan {min_year} och {max_year}.")
        except ValueError:
            print("Vänligen ange ett giltigt årtal.")

START_YEAR = get_year("Ange startår: ")
END_YEAR = get_year(f"Ange slutår (minst {START_YEAR}): ", min_year=START_YEAR)

def get_fuel_type():
    while True:
        choice = input("Drivmedel (1: Bensin, 2: Diesel, 3: El, 4: Hybrid): ")
        if choice == "1":
            return "1"
        elif choice == "2":
            return "2"
        elif choice == "3":
            return "3"
        elif choice == "4":
            return "9999"
        else:
            print("Vänligen ange ett giltigt drivmedel.")

FUEL = get_fuel_type()

def get_gearbox_type():
    while True:
        choice = input("Växellåda (1: Automat, 2: Manuell): ")
        if choice == "1":
            return "1000"
        elif choice == "2":
            return "5"
        else:
            print("Vänligen ange en giltig växellåda.")

GEARBOX = get_gearbox_type()

URL = "https://www.car.info/sv-se/[MAKE]/[MODEL]/classifieds?fuel=[FUEL]&trans=[GEARBOX]&year_min=[START YEAR]&year_max=[END YEAR]&seller=st_private"

URL = URL.replace("[MAKE]", MAKE)\
       .replace("[MODEL]", MODEL)\
       .replace("[START YEAR]", str(START_YEAR))\
       .replace("[END YEAR]", str(END_YEAR))\
       .replace("[FUEL]", FUEL)\
       .replace("[GEARBOX]", GEARBOX)

def parse_site(url):
    try:
        page = requests.get(url, timeout=10)
        page.raise_for_status()
        soup = BeautifulSoup(page.content, "html.parser")
        return soup
    except requests.RequestException as e:
        raise Exception(f"Failed to fetch the page: {str(e)}")

def get_all_ads(url):
    print(f"Fetching from URL: {url}")
    soup = parse_site(url)
    ad_elements = soup.find_all("tr", class_="classified_item list-row position-relative")
    
    print(f"Found {len(ad_elements)} elements")
    ads = []
    for ad in ad_elements:
        
        ad_url_element = ad.find("a", class_="classified_url flex-grow-1 fw-bold text-truncate rec_name")
        ad_url = ad_url_element["href"] if ad_url_element else "N/A"
        
        ad_price_element = ad.find("div", class_="d-flex justify-content-end")
        ad_price = ad_price_element.text if ad_price_element else "N/A"
        
        ad_mileage_element = ad.find("td", class_="d-none d-lg-table-cell text-nowrap")
        ad_mileage = ad_mileage_element.text if ad_mileage_element else "N/A"
        
        ads.append({
            'url': ad_url,
            'price': ad_price,
            'mileage': ad_mileage
        })
    
    return ads

ads = get_all_ads(URL)

with open("listings.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["url", "price", "mileage"])
    writer.writeheader()
    writer.writerows(ads)

print(f"Ads have been saved to listings.csv")

def clean_number(text):
    """Remove non-numeric characters and return the cleaned number."""
    cleaned = ''.join(filter(str.isdigit, text))
    return int(cleaned) if cleaned else 0

with open("listings.csv", mode="r", encoding="utf-8") as infile, \
     open("listings_cleaned.csv", mode="w", newline="", encoding="utf-8") as outfile:
    
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for row in reader:
        row['price'] = clean_number(row['price'])
        row['mileage'] = clean_number(row['mileage'])
        writer.writerow(row)

print("Cleaned data has been saved to listings_cleaned.csv")

prices = np.array([clean_number(ad['price']) for ad in ads], dtype=np.float64)
mileages = np.array([clean_number(ad['mileage']) for ad in ads], dtype=np.float64)

X = mileages.reshape(-1, 1)
y = prices.reshape(-1, 1)

model = LinearRegression()
model.fit(X, y)

predicted_prices = model.predict(X)
deviations = (y - predicted_prices) / predicted_prices

plt.figure(figsize=(10, 5))

for i in range(len(mileages)):
    if deviations[i] <= -0.15:
        color = 'green'
    elif deviations[i] >= 0.15:
        color = 'red'
    else:
        color = 'blue'
    
    point = plt.scatter(mileages[i], prices[i], color=color, alpha=0.5)

plt.plot(X, predicted_prices, color='blue', label='Regressionslinje')

plt.xlabel('Miltal (mil)')
plt.ylabel('Pris (kr)')
plt.title(f'Regressionsanalys för {MAKE} {MODEL}')

r2_score = model.score(X, y)
plt.text(0.05, 0.95, f'R² = {r2_score:.2f}', 
         transform=plt.gca().transAxes, 
         bbox=dict(facecolor='white', alpha=0.8))

plt.savefig("regression_plot.png")
print("Regressionsanalys have been saved as regression_plot.png")

results = []
for i in range(len(prices)):
    under_over_valued = (predicted_prices[i][0] - prices[i])
    potential_roi = under_over_valued / prices[i] if prices[i] != 0 else 0
    results.append({
        'url': ads[i]['url'],
        'price': prices[i],
        'mileage': mileages[i],
        'under_over_valued': under_over_valued,
        'potential_roi': potential_roi
    })

with open("listings_results.csv", mode="w", newline="", encoding="utf-8") as results_file:
    fieldnames = ["url", "price", "mileage", "under_over_valued", "potential_roi"]
    writer = csv.DictWriter(results_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print("Results have been saved to listings_results.csv")

plt.show()