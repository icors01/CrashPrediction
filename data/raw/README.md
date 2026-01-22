# How to download full crash data

Only some sample crash data (August 2025) is included in the repository, as the entire data set would be too large to easily fit

To download the full data set that we will be using:

go to [Minnesota Road Safety Information Center's website](https://roadsafetyinfocenter.mn.gov/map/information/crashes)

Select a start date of 01/01/2016 and an end date of 09/30/2025 (there's data beyond September 2025, but for consistency this is the end date we will use)

Click the download button in the top left of the map

Deselect everything besides "All Crashes" and then click Export Data

Once the data finishes exporting, click the green bar again, and download the file

Rename the file to "all_crashes.csv" and place it in the /data/raw folder


# How to download weather data

Go to [National Centers for Environmental Information](https://www.ncei.noaa.gov/cdo-web/search?datasetid=GHCND)

Select Weather Observation Type as Daily Summaries, Date Range as 2016-01-01 to 2025-09-30, Search for Stations, 

For search term, add these three stations to cart 
- INTERNATIONAL FALLS INTERNATIONAL AIRPORT, MN US 
- MINNEAPOLIS ST. PAUL INTERNATIONAL AIRPORT, MN US
- ROCHESTER INTERNATIONAL AIRPORT, MN US

Select Daily CSV Output in Cart, make sure date range is correct, and continue

Select Station Name, Make Units Standard, Select Data Types 'Precipiation', 'Air Temperature', and 'Weather Type'

Enter email address and NOAA will send the data file to your email

Download file as "weather_data.csv" and add to /data/raw folder
