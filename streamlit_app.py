import streamlit as st
import requests
from bs4 import BeautifulSoup
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import csv
from io import StringIO

# Titel och introduktion
st.title("Bilvärderingsapp")
st.write("Denna app hämtar bilannonser från Car.info, utför regressionsanalys och visar resultaten.")
st.write("Ange sökparametrar i sidopanelen för att börja. Dubbelkolla att du matar in rätt parametrar enligt Car.info's filter (https://www.car.info/sv-se/filter).")

# Sidopanel för användarinmatning
st.sidebar.header("Sökparametrar")
make = st.sidebar.text_input("Ange bilmärke", value="Volvo")
model = st.sidebar.text_input("Ange bilmodell", value="V70")
start_year = st.sidebar.number_input("Startår", min_value=1920, max_value=2025, value=2010)
end_year = st.sidebar.number_input("Slutår", min_value=start_year, max_value=2025, value=2020)
fuel_type = st.sidebar.selectbox("Bränsletyp", ["Bensin", "Diesel", "El", "Hybrid"])
gearbox_type = st.sidebar.selectbox("Växellåda", ["Automat", "Manuell"])

# Mappa bränsle- och växellådstyper till query-parametrar
fuel_map = {"Bensin": "1", "Diesel": "2", "El": "3", "Hybrid": "9999"}
gearbox_map = {"Automat": "1000", "Manuell": "5"}
fuel = fuel_map[fuel_type]
gearbox = gearbox_map[gearbox_type]

# Konstruera URL
url = f"https://www.car.info/sv-se/{make}/{model}/classifieds?fuel={fuel}&trans={gearbox}&year_min={start_year}&year_max={end_year}&seller=st_private"

# Funktion för att hämta och analysera annonser
def get_all_ads(url):
    try:
        page = requests.get(url, timeout=10)
        page.raise_for_status()
        soup = BeautifulSoup(page.content, "html.parser")
        ad_elements = soup.find_all("tr", class_="classified_item list-row position-relative")
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
    except Exception as e:
        st.error(f"Misslyckades med att hämta annonser: {e}")
        return []

# Hämta annonser och utför regressionsanalys
st.write(f"Hämtar annonser för {make} {model}...")
ads = get_all_ads(url)

if ads:
    st.success(f"Hämtade {len(ads)} annonser!")

    # Rensa och bearbeta data
    def clean_number(text):
        cleaned = ''.join(filter(str.isdigit, text))
        return int(cleaned) if cleaned else 0

    prices = np.array([clean_number(ad['price']) for ad in ads], dtype=np.float64)
    mileages = np.array([clean_number(ad['mileage']) for ad in ads], dtype=np.float64)

    X = mileages.reshape(-1, 1)
    y = prices.reshape(-1, 1)

    # Träna regressionsmodellen
    model = LinearRegression()
    model.fit(X, y)
    predicted_prices = model.predict(X)

    # Beräkna avvikelser och potentiell ROI
    results = []
    for i in range(len(prices)):
        under_over_valued = round(predicted_prices[i][0] - prices[i])
        potential_roi = round(under_over_valued / prices[i], 2) if prices[i] != 0 else 0
        results.append({
            'url': ads[i]['url'],
            'price': prices[i],
            'mileage': mileages[i],
            'under_over_valued': under_over_valued,
            'potential_roi': potential_roi
        })

    # Konvertera resultat till en DataFrame för visning
    results_df = pd.DataFrame(results)

    # Färgkoda annonser baserat på avvikelse
    for i in range(len(mileages)):
        if results_df['under_over_valued'][i] <= -0.15 * prices[i]:
            color = 'green'
        elif results_df['under_over_valued'][i] >= 0.15 * prices[i]:
            color = 'red'
        else:
            color = 'blue'
        plt.scatter(mileages[i], prices[i], color=color, alpha=0.5)    

    # Rita regressionsdiagram
    plt.figure(figsize=(10, 5))
    plt.scatter(mileages, prices, color='blue', alpha=0.5, label="Annonser")
    plt.plot(X, predicted_prices, color='blue', label="Regressionslinje")
    plt.xlabel("Miltal (mil)")
    plt.ylabel("Pris (SEK)")
    plt.title(f"Regressionsanalys för {make} {model}")
    plt.legend()

    st.write("### Resultat från regressionsanalys")
    st.write("Regressionslinjen visar det förväntade priset baserat på miltal där R^2 värdet visar hur tillförlitlig modellen är.")
    st.write("Annonser markerade i grönt är undervärderade, medan de i rött är övervärderade och de i blått är normalpriser.")
    st.pyplot(plt)

    # Visa resultat i tabell
    st.write("### Resultat från annonser")
    st.write("Tabellen nedan visar annonser med deras URL, pris, miltal, avvikelse och potentiell ROI.")
    st.dataframe(results_df)
else:
    st.warning("Inga annonser hittades.")