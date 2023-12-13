# DASHBOARD FOR THE BOOKING and ACTUAL
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
import numpy as np
import calendar

import requests
import os
import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt

load_dotenv()

st.set_page_config(
    page_title="kidzapp dashboard",  # Set your app title here
    page_icon="📊",  # You can also set an emoji or an image as an icon
    layout="wide",  # Optional: Use 'wide' layout
    initial_sidebar_state="expanded",  # Optional: Expand or collapse the sidebar
    menu_items={
        "Get Help": "https://www.extremelycoolapp.com/help",
        "Report a bug": "https://www.extremelycoolapp.com/bug",
        "About": "# This is a very cool app!",  # You can also set your own About text or link
    },
)


def create_gauge_with_color_and_data(week_values, total_value=7000):
    # Calculate the total of the weekly values
    total_week_values = sum(week_values)
    # Calculate percentages based on individual weekly values
    percentages = [round((value / total_value) * 100, 2) for value in week_values]
    cumulative_values = [sum(week_values[: i + 1]) for i in range(len(week_values))]
    percentages = [round((value / total_value) * 100, 2) for value in cumulative_values]

    # Create gauge figure with RGBA color based on individual weekly values
    fig = go.Figure(
        go.Indicator(
            number=dict(font=dict(size=20)),
            mode="gauge+number",
            value=total_week_values,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Actual Weekly Progress (%)", "font": {"size": 24}},
            gauge={
                "axis": {"range": [None, total_value], "tickcolor": "darkblue"},
                "bar": {"color": "darkblue"},
                "steps": [
                    {
                        "range": [0, week_values[0]],
                        "color": f"rgba(0, 255, 0, {percentages[1]/100})",
                    },
                    {
                        "range": [week_values[0], sum(week_values[:2])],
                        "color": f"rgba(0, 0, 255, {percentages[1]/100})",
                    },
                    {
                        "range": [sum(week_values[:2]), sum(week_values[:3])],
                        "color": f"rgba(255, 0, 0, {percentages[2]/100})",
                    },
                    {
                        "range": [sum(week_values[:3]), sum(week_values)],
                        "color": f"rgba(255, 255, 0, {percentages[3]/100})",
                    },
                ],
            },
        )
    )

    # Adding annotations for weekly values
    for i, value in enumerate(week_values):
        fig.add_annotation(
            x=0.5,
            y=0.5 - (0.1 * i),
            xref="paper",
            yref="paper",
            text=f"Week {i+1}: {value} ({percentages[i]}%)",
            showarrow=False,
        )

    # Calculate trend arrow and color based on the last two weeks
    if len(week_values) > 1:
        trend_arrow = "↑" if week_values[-1] >= week_values[-2] else "↓"
        trend_color = "green" if trend_arrow == "↑" else "red"
    else:
        trend_arrow = ""
        trend_color = "black"

    # Adding trend arrow annotation
    fig.add_annotation(
        x=0.5,
        y=-0.2,
        xref="paper",
        yref="paper",
        text=f"Trend: {trend_arrow}",
        font={"color": trend_color, "size": 18},
        showarrow=False,
    )

    fig.update_layout(
        paper_bgcolor="lavender", font={"color": "darkblue", "family": "Arial"}
    )
    return fig


def test_view():
    image = "kidzapp-logo.png"  # Replace with the actual path to your image
    st.image(
        image,
        width=200,  # Adjust the width to your desired size
    )
    # Streamlit app
    st.title("Kidzapp Weekly Dashboard")

    # Weekly values
    week_values = [1201, 1024, 894, 913]  # Values for each week
    gauge_chart = create_gauge_with_color_and_data(week_values)

    st.plotly_chart(gauge_chart)

    # Display legend with trend arrows
    st.write("### Legend")
    for i, value in enumerate(week_values):
        trend_arrow = "↑" if i == 0 or week_values[i] >= week_values[i - 1] else "↓"
        trend_color = "green" if trend_arrow == "↑" else "red"
        colored_text = f'<span style="color: {trend_color};">Week {i+1}: {value} ({trend_arrow})</span>'
        st.markdown(colored_text, unsafe_allow_html=True)

    # Data
    week_values = [1201, 1024, 894, 913]
    weekly_target = [1166, 1166, 1166, 1166]
    lastyear_target = [831, 708, 665, 605]

    # Create a DataFrame for plotting
    df = pd.DataFrame(
        {
            "Week": range(1, len(week_values) + 1),
            "Weekly Target": weekly_target,
            "Actual Values": week_values,
            "Last Year Target": lastyear_target,
        }
    )

    # Create a clustered bar chart
    fig = px.bar(
        df,
        x="Week",
        y=["Actual Values", "Weekly Target", "Last Year Target"],
        labels={"value": "Values"},
        title="Comparison of Actual to Weekly and Last Year Booking Count",
        barmode="group",
    )  # Use 'group' for clustered bars
    fig.update_layout(xaxis_title="Week", yaxis_title="Values")
    st.plotly_chart(fig)


def trend_indicator_view():
    # Define the data with padding for missing values
    data = {
        "Provider": [
            "Rayna",
            "Cheeky Monkeys",
            "Orange Wheels",
            "Bounce",
            "Emaar (Main)",
            "Ready Set Go",
            "The Farm",
            "Air Maniax",
            "Playtorium",
            "Kids HQ",
            "Splash N Party",
            "Dubai Garden Glow",
            "OliOli",
            "Woo-hoo",
            "Adventure Parx + Cafe",
            "Dubai Parks and Resorts",
            "Aqua Parks Leisure",
            "SupperClub",
            "Emirates Park Zoo",
            "Bricobilandia",
            "Piccoli",
            "Ready Steady Go",
            "Marinelys Babysitting Center",
            "Fiafia",
            "Tour Dubai (Main)",
            "Tr88house",
            "Kids Unlimited",
            "Dare Park",
            "Gymnastex",
            "Splash Island",
            "Museum of Illusions",
            "Cuckoo's",
            "Kidzapp",
            "Aventura",
            "Tommy Life Kids Amusement Arcade",
            "Kids Hub Entertainment",
            "Molly Coddle Baby Spa",
            "Playville Kids Amusement Arcade",
            "Kids Zone (Main)",
            "3D World Selfie Museum Dubai",
            "Adventureland",
            "Bricks 4 Fun",
            "La La Land",
            "The National Aquarium",
            "Prosportsuae",
            "Chillout Lounge",
            "Gogo Village",
            "Rose Ballet",
            "Suwaidi Pearl Farm",
            "Shurooq (Main)",
            "Beitfann",
            "Toda Dubai",
            "Circuit X (Main)",
            "IMG Worlds of Adventures",
            "Swimaholic",
            "Sahara Marine",
            "Little Champions Club",
            "Priohub",
            "Swissotel Al Ghurair",
            "Doodle Kids Play",
            "Bab Al Nojoum",
        ],
        "Week 1": [
            255,
            167,
            94,
            94,
            84,
            54,
            46,
            46,
            33,
            62,
            24,
            26,
            19,
            20,
            14,
            21,
            19,
            15,
            10,
            10,
            7,
            12,
            5,
            3,
            3,
            8,
            15,
            5,
            4,
            3,
            1,
            1,
            2,
            0,
            2,
            2,
            2,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            1,
            0,
            1,
        ],
        "Week 2": [
            241,
            117,
            115,
            80,
            57,
            49,
            41,
            36,
            37,
            15,
            24,
            24,
            22,
            13,
            21,
            12,
            12,
            10,
            11,
            9,
            8,
            7,
            5,
            8,
            3,
            4,
            0,
            6,
            3,
            1,
            1,
            3,
            3,
            3,
            0,
            3,
            3,
            2,
            0,
            0,
            3,
            3,
            0,
            0,
            1,
            0,
            1,
            0,
            2,
            1,
            1,
            0,
            0,
            1,
        ],
        "Week 3": [
            209,
            104,
            65,
            53,
            59,
            36,
            48,
            34,
            29,
            18,
            39,
            20,
            17,
            19,
            9,
            8,
            10,
            13,
            14,
            13,
            12,
            4,
            8,
            6,
            5,
            2,
            0,
            0,
            5,
            7,
            3,
            3,
            0,
            0,
            0,
            1,
            2,
            1,
            2,
            0,
            2,
            0,
            1,
            3,
            3,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
        ],
        "Week 4": [
            232,
            120,
            94,
            55,
            41,
            33,
            30,
            34,
            23,
            21,
            16,
            22,
            22,
            23,
            17,
            15,
            13,
            9,
            11,
            10,
            9,
            5,
            8,
            3,
            6,
            3,
            1,
            2,
            2,
            5,
            5,
            3,
            4,
            4,
            0,
            1,
            1,
            1,
            0,
            1,
            1,
            0,
            2,
            1,
            1,
            0,
            1,
            1,
            1,
            1,
            0,
            1,
        ],
    }

    import streamlit as st
    import pandas as pd
    import plotly.express as px

    # Define the data with padding for missing values
    # (Same data as before)

    # Ensure all weeks have the same length by padding with zeros
    max_week_length = max(len(data[week]) for week in data)
    for week in data:
        data[week] += [0] * (max_week_length - len(data[week]))

    # Create a DataFrame with numerical columns
    df = pd.DataFrame(data)

    # Convert numerical columns to integers
    numeric_columns = df.columns[1:]  # Exclude the "Venue" column
    df[numeric_columns] = df[numeric_columns].astype(int)

    # Create a Streamlit app
    st.title("Weekly Data with Trend Indicator")

    # Reset the index to include the "Venue" column
    df.reset_index(drop=True, inplace=True)

    # Display the DataFrame with trend indicators
    prev_week = df.columns[-2]  # Get the name of the previous week's column
    current_week = df.columns[-1]  # Get the name of the current week's column

    # Function to calculate trend indicator (Up or Down)
    def calculate_trend_indicator(row):
        if row[current_week] > row[prev_week]:
            return "Up"
        elif row[current_week] < row[prev_week]:
            return "Down"
        else:
            return ""

    # Apply the function to each row to calculate the trend indicator
    df["Trend"] = df.apply(calculate_trend_indicator, axis=1)

    # Set the "Provider" column as the index
    df.set_index("Provider", inplace=True)

    # Create a bar chart of the top 10 venues for the current week
    top_10_venues = df.nlargest(10, current_week)
    fig = px.bar(
        top_10_venues,
        x=top_10_venues.index,
        y=current_week,
        title="Top 10 Providers for Current Week",  # Updated title
    )
    fig.update_xaxes(title="Provider")
    fig.update_yaxes(title=current_week)
    # Display the table with trend indicators
    # st.write(df)
    st.dataframe(df, use_container_width=True)
    st.plotly_chart(fig)


# ================================================================ KEYWORD VIEWS =================================================================


@st.cache_data
def fetch_keywords_from_clevertap(from_date, to_date):
    # Constants for the event
    EVENT_NAME = "Searched"

    headers = {
        "X-CleverTap-Account-Id": st.secrets["CLEVERTAP_ACCOUNT_ID"],
        "X-CleverTap-Passcode": st.secrets["CLEVERTAP_PASSCODE"],
        "Content-Type": "application/json",
    }

    # Construct the groups object with proper property_type
    groups = {
        "eventPropertyProductViews": {
            "property_type": "event_properties",
            "name": "Keyword",
            "top_n": 26,
            "order": "desc",
        },
    }

    # Data payload as a dictionary
    data = {
        "event_name": EVENT_NAME,
        "common_profile_properties": {
            "geo_fields": [
                ["United Arab Emirates"],
            ]
        },
        "from": int(from_date),
        "to": int(to_date),
        "groups": groups,
    }

    # Making the POST request to CleverTap
    response = requests.post(
        "https://api.clevertap.com/1/counts/top.json",
        headers=headers,
        json=data,  # Sends the dictionary as a JSON-formatted string
    )
    print(data)
    print(response.text)
    if response.status_code == 200:
        json_obj = json.loads(response.text)
        count_value = json_obj["req_id"]

        response_f = requests.post(
            "https://api.clevertap.com/1/counts/top.json?req_id=" + count_value,
            headers=headers,
            data=data,
        )

        if response_f.status_code == 200:
            json_response = response_f.json()
            keywords = json_response.get("eventPropertyProductViews", {}).get("STR", {})
            # print(keywords, "keywords")
            return keywords
        else:
            st.error("Failed to fetch detailed data from CleverTap")
            return []
    else:
        st.error("Failed to initiate request to CleverTap")
        return []


def display_wordcloud(keywords):
    # Exclude '-1' from keywords
    if "-1" in keywords:
        del keywords["-1"]

    # Sort and keep top 26 keywords
    sorted_keywords = dict(
        sorted(keywords.items(), key=lambda item: item[1], reverse=True)[:26]
    )

    wordcloud = WordCloud(
        width=800, height=400, background_color="white"
    ).generate_from_frequencies(sorted_keywords)
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    st.pyplot(plt)


def convert_to_dataframe(keywords):
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(list(keywords.items()), columns=["Keyword", "Count"])

    # Filter out rows where Keyword is '-1'
    df = df[df["Keyword"] != "-1"]
    # Sort and keep top 26 keywords
    sorted_keywords = dict(
        sorted(keywords.items(), key=lambda item: item[1], reverse=True)[:26]
    )

    # Convert the sorted dictionary to a DataFrame
    df = pd.DataFrame(list(sorted_keywords.items()), columns=["Keyword", "Count"])

    return df


def to_csv(df):
    # Convert DataFrame to CSV
    return df.to_csv().encode("utf-8")


# Main function to handle the Streamlit view
def keyword_view():
    st.title("CleverTap Keyword Analysis")

    # Date pickers for last week
    st.subheader("Last Week")
    start_date_last = st.date_input("Start Date (Last Week)")
    end_date_last = st.date_input("End Date (Last Week)")

    # Date pickers for current week
    st.subheader("Current Week")
    start_date_current = st.date_input("Start Date (Current Week)")
    end_date_current = st.date_input("End Date (Current Week)")

    if st.button("Fetch Keywords"):
        # Fetch and display keywords for last week
        from_date_last = start_date_last.strftime("%Y%m%d")
        to_date_last = end_date_last.strftime("%Y%m%d")
        keywords_last = fetch_keywords_from_clevertap(from_date_last, to_date_last)

        if keywords_last:
            st.subheader(
                f"Keywords for Last Week ({start_date_last} to {end_date_last})"
            )
            display_wordcloud(keywords_last)
            df_last = convert_to_dataframe(keywords_last)
            st.dataframe(df_last, use_container_width=True)
            csv_last = to_csv(df_last)
            # For the last week data download button
            st.download_button(
                label="Download Last Week Data as CSV",
                data=csv_last,
                file_name="last_week_keywords.csv",
                mime="text/csv",
                key="download_last_week",  # Unique key for this button
            )

        # Fetch and display keywords for current week
        from_date_current = start_date_current.strftime("%Y%m%d")
        to_date_current = end_date_current.strftime("%Y%m%d")
        keywords_current = fetch_keywords_from_clevertap(
            from_date_current, to_date_current
        )

        if keywords_current:
            st.subheader(
                f"Keywords for Current Week ({start_date_current} to {end_date_current})"
            )
            display_wordcloud(keywords_current)
            df_current = convert_to_dataframe(keywords_current)
            st.dataframe(df_current, use_container_width=True)
            csv_current = to_csv(df_current)
            # For the current week data download button

            st.download_button(
                label="Download Current Week Data as CSV",
                data=csv_current,
                file_name="current_week_keywords.csv",
                mime="text/csv",
                key="download_current_week",  # Unique key for this button
            )


# ================================================================ Experience / PRODUCT VIEWS =================================================================


@st.cache_data
def fetch_product_views(from_date, to_date):
    EVENT_NAME = "Product Viewed"

    headers = {
        "X-CleverTap-Account-Id": st.secrets["CLEVERTAP_ACCOUNT_ID"],
        "X-CleverTap-Passcode": st.secrets["CLEVERTAP_PASSCODE"],
        "Content-Type": "application/json",
    }

    groups = {
        "eventPropertyProductViews": {
            "property_type": "event_properties",
            "name": "Product Name",
            "top_n": 26,
            "order": "desc",
        },
    }

    data = {
        "event_name": EVENT_NAME,
        "common_profile_properties": {
            "geo_fields": [
                ["United Arab Emirates"],
            ]
        },
        "from": int(from_date),
        "to": int(to_date),
        "groups": groups,
    }

    response = requests.post(
        "https://api.clevertap.com/1/counts/top.json",
        headers=headers,
        json=data,
    )

    if response.status_code == 200:
        json_obj = json.loads(response.text)
        count_value = json_obj["req_id"]

        response_f = requests.post(
            f"https://api.clevertap.com/1/counts/top.json?req_id={count_value}",
            headers=headers,
            json=data,
        )

        if response_f.status_code == 200:
            return response_f.json()
        else:
            st.error("Failed to fetch detailed data from CleverTap")
            return None
    else:
        st.error("Failed to initiate request to CleverTap")
        return None


from sqlalchemy import create_engine, text
import pandas as pd


def get_booking_counts(start_date, end_date):
    engine = get_connection()  # Use your function to get the connection

    # Adjusted query to fetch booking counts for all experiences
    query = text(
        """
        SELECT
        ee.title,
        COUNT(bb.id) AS booking_count
        FROM
        booking_booking bb
        JOIN
        experiences_experience ee ON bb.experience_id = ee.id
        WHERE
        bb.payment_status = 'CAPTURED'
        AND bb.created_at >= :start_date AND bb.created_at < :end_date
        GROUP BY
        ee.title
        ORDER BY
        booking_count DESC;
        """
    )

    # Execute the SQL query using the connection
    with engine.connect() as conn:
        result = conn.execute(
            query, {"start_date": start_date, "end_date": end_date}
        ).fetchall()

    df = pd.DataFrame(result, columns=["Experience Title", "Booking Count"])
    return df


def display_horizontal_bar_chart(df, title):
    # Sort the DataFrame in ascending order for the chart
    df_chart = df.sort_values(by="Count", ascending=True)

    # Using Plotly to create a horizontal bar chart
    fig = px.bar(
        df_chart,
        y="Product Name",
        x="Count",
        orientation="h",
        color="Count",
        color_continuous_scale="Blues",
        height=800,
    )
    fig.update_layout(
        title_text=title,
        xaxis_title="Count",
        yaxis_title="Product Name",
    )
    st.plotly_chart(fig, use_container_width=True)


def process_data(data):
    if data:
        product_data = data.get("eventPropertyProductViews", {}).get("STR", {})
        df = pd.DataFrame(product_data.items(), columns=["Product Name", "Count"])
        df = df[df["Product Name"] != "-1"].sort_values(by="Count", ascending=False)
        return df
    return pd.DataFrame()


from fuzzywuzzy import process


def calculate_conversion_rate(booking_df, product_view_df):
    product_names = product_view_df["Product Name"].tolist()  # Ensure this is a list

    # Create a mapping of similar names
    mapping = {}
    for title in booking_df["Experience Title"]:
        # Find the most similar product name for each booking title
        match = process.extractOne(title, product_names)
        if match:
            best_match, similarity = match
            if similarity > 80:  # Adjust threshold as needed
                mapping[title] = best_match

    # Create new columns in booking_df for product view count and conversion rate
    booking_df["Product View Count"] = 0
    booking_df["Conversion Rate (%)"] = 0

    for title, match in mapping.items():
        if match in product_view_df["Product Name"].values:
            view_count = product_view_df.loc[
                product_view_df["Product Name"] == match, "Count"
            ].values[0]
            booking_count = booking_df.loc[
                booking_df["Experience Title"] == title, "Booking Count"
            ].values[0]
            conversion_rate = (booking_count / view_count) * 100 if view_count else 0
            booking_df.loc[
                booking_df["Experience Title"] == title, "Product View Count"
            ] = view_count
            booking_df.loc[
                booking_df["Experience Title"] == title, "Conversion Rate (%)"
            ] = conversion_rate

    return booking_df


def experience_view():
    st.title("CleverTap Experience View Analysis")

    # Date pickers for last week
    st.subheader("Last Week")
    start_date_last = st.date_input("Start Date (Last Week)", key="start_last")
    end_date_last = st.date_input("End Date (Last Week)", key="end_last")

    # Date pickers for current week
    st.subheader("Current Week")
    start_date_current = st.date_input("Start Date (Current Week)", key="start_current")
    end_date_current = st.date_input("End Date (Current Week)", key="end_current")

    if st.button("Fetch Experience Views"):
        # Fetch and display data for last week
        from_date_last = start_date_last.strftime("%Y%m%d")
        to_date_last = end_date_last.strftime("%Y%m%d")
        data_last = fetch_product_views(from_date_last, to_date_last)
        df_last = process_data(data_last)

        if not df_last.empty:
            st.subheader("Experience View - Last Week")
            # display the view
            st.dataframe(df_last.head(25), use_container_width=True)
            display_horizontal_bar_chart(
                df_last.head(25), "Top 25 Product Views - Last Week"
            )

        # Fetch and display data for current week
        from_date_current = start_date_current.strftime("%Y%m%d")
        to_date_current = end_date_current.strftime("%Y%m%d")
        data_current = fetch_product_views(from_date_current, to_date_current)
        df_current = process_data(data_current)

        if not df_current.empty:
            st.subheader("Experience View - Current Week")
            # display the views
            st.dataframe(df_current.head(25), use_container_width=True)
            display_horizontal_bar_chart(
                df_current.head(25), "Top 25 Product Views - Current Week"
            )

        # Assuming you now have df_last and df_current as the top 25 product views
        # Extract the product titles
        product_titles_last_week = df_last["Product Name"].tolist()
        product_titles_current_week = df_current["Product Name"].tolist()

        # Fetch booking counts for all experiences
        booking_counts_last_week = get_booking_counts(from_date_last, to_date_last)
        booking_counts_current_week = get_booking_counts(
            from_date_current, to_date_current
        )

        # Display booking counts in the Streamlit app
        st.subheader("Booking Counts - Last Week")
        st.dataframe(booking_counts_last_week, use_container_width=True)

        st.subheader("Booking Counts - Current Week")
        st.dataframe(booking_counts_current_week, use_container_width=True)

        # Sort df_last by 'Product View Count' in descending order and take the top 25
        sorted_df_last = df_last.sort_values(by='Count', ascending=False).head(25)
        sorted_df_current = df_current.sort_values(by='Count', ascending=False).head(25)


        # === conversion part
        # Calculate and display conversion rates
        conversion_rates_last_week = calculate_conversion_rate(
            booking_counts_last_week, sorted_df_last
        )
        conversion_rates_current_week = calculate_conversion_rate(
            booking_counts_current_week, sorted_df_current
        )

        # Display booking counts with conversion rates in the Streamlit app
        st.subheader("Booking Counts with Conversion Rates - Last Week")
        st.dataframe(conversion_rates_last_week, use_container_width=True)

        st.subheader("Booking Counts with Conversion Rates - Current Week")
        st.dataframe(conversion_rates_current_week, use_container_width=True)


# ============================================================================= the viewer =================================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import sqlalchemy


# Function to create a database connection
def get_connection():
    # Replace with your actual database details
    db_type = st.secrets["db_type"]
    username = st.secrets["username"]
    password = st.secrets["password"]
    database_name = st.secrets["database_name"]
    host = st.secrets["host"]

    return sqlalchemy.create_engine(
        f"{db_type}://{username}:{password}@{host}/{database_name}"
    )


@st.cache_data
def create_query(start_date, end_date, previous_year=False):
    query_parts = ["SELECT pp.name AS provider_name"]

    # Calculate the start and end dates for the previous year if required
    if previous_year:
        start_date = start_date.replace(year=start_date.year - 1)
        end_date = end_date.replace(year=end_date.year - 1)

    # Initialize the current_date as start_date
    current_date = start_date
    week_count = 1

    while current_date <= end_date:
        week_end = current_date + datetime.timedelta(days=6)
        if week_end > end_date:
            week_end = end_date

        # Adding each COUNT clause with a leading comma
        query_parts.append(
            f", COUNT(CASE WHEN bb.created_at BETWEEN '{current_date}' AND '{week_end}' THEN 1 END) AS week_{week_count}_count"
        )

        # Update current_date to the next period
        current_date = week_end + datetime.timedelta(days=1)
        week_count += 1

    # Adding the remaining part of the SQL query
    query_parts.append(
        f"""
    FROM
        booking_booking bb
    JOIN
        experiences_experience ee ON bb.experience_id = ee.id
    JOIN
        provider_provider pp ON ee.provider_id = pp.id
    JOIN
        core_city cc ON ee.city_id = cc.id
    JOIN
        core_country co ON cc.country_id = co.id
    WHERE
        bb.payment_status = 'CAPTURED'
        AND co.name = 'United Arab Emirates'
        AND bb.created_at BETWEEN '{start_date}' AND '{end_date}'
    GROUP BY
        pp.name
    ORDER BY
        pp.name
    """
    )

    return " ".join(query_parts)


def analyze_booking_view():
    st.title("Weekly Booking Analysis")

    # Allow the user to select from the years 2022 and 2023 only
    analysis_year = st.selectbox("Select the year for analysis", [2023, 2022, 2021])

    # Calculate dates for the selected year
    jan_1 = datetime.date(analysis_year, 1, 1)
    dec_31 = datetime.date(analysis_year, 12, 31)

    # Set default dates for the date input
    today = datetime.date.today()
    default_start_date = today if jan_1 <= today <= dec_31 else jan_1
    default_end_date = default_start_date + datetime.timedelta(days=6)

    # User selects vacation dates for the selected year
    vacation_dates = st.date_input(
        "Select your dates",
        (default_start_date, default_end_date),
        min_value=jan_1,
        max_value=dec_31,
        format="MM.DD.YYYY",
    )

    # Check if the returned value is a tuple with two dates
    if not isinstance(vacation_dates, tuple) or len(vacation_dates) != 2:
        st.warning("Please complete the date range.")
        return  # Exit the function

    # Extract start and end dates from the selected range
    start_date, end_date = vacation_dates

    # Check if start and end dates are in the same month
    if start_date.month != end_date.month or start_date.year != end_date.year:
        st.warning(
            "Start and End dates must be in the same month. Please adjust the dates."
        )
        return  # Exit the function if dates are not in the same month

    if st.button("Analyze Bookings"):
        # Queries for current and previous year
        query_current_year = create_query(start_date, end_date)
        query_previous_year = create_query(start_date, end_date, previous_year=True)

        # Database connection
        conn = get_connection()

        # Retrieve data for current and previous year
        data_current_year = pd.read_sql(query_current_year, conn)
        data_previous_year = pd.read_sql(query_previous_year, conn)

        # Define weeks
        weeks = ["week_1_count", "week_2_count", "week_3_count", "week_4_count"]

        # Ensure all week columns exist and set default values if missing
        for week in weeks:
            if week not in data_current_year.columns:
                data_current_year[week] = 0
            if week not in data_previous_year.columns:
                data_previous_year[week] = 0

        # Define colors
        colors = ["green", "darkorange", "lightgray", "orange"]

        # Specific values to add as a stacked column (weekly target)
        specific_values_wk_target = np.array(
            [1166, 1166, 1166, 1166]
        )  # Converted to numpy array

        # Plotting
        fig, ax = plt.subplots()

        # Plotting function for each data set
        def plot_data(data, ypos):
            total_bookings_per_week = data[weeks].sum()
            cumulative = total_bookings_per_week.cumsum()

            for i in range(len(weeks)):
                left = 0 if i == 0 else cumulative[i - 1]
                bar_width = total_bookings_per_week[i]
                bar = ax.barh(
                    ypos,
                    bar_width,
                    left=left,
                    color=colors[i],
                    label=weeks[i] if ypos == 2 else "",
                )

                # Add text label inside each bar segment
                text_x_position = left + bar_width / 2
                ax.text(
                    text_x_position,
                    ypos,
                    f"{bar_width}",
                    va="center",
                    ha="center",
                    color="white",
                )

        # Plotting data for current year, previous year, and weekly target
        plot_data(
            pd.DataFrame(
                {wk: [specific_values_wk_target[i]] for i, wk in enumerate(weeks)}
            ),
            2,
        )
        plot_data(data_current_year, 1)
        plot_data(data_previous_year, 0)

        # Set the chart title, labels, and legend
        month_name = calendar.month_name[start_date.month]
        ax.set_title(
            f"Cumulative Total Bookings by Week for {month_name} {start_date.year}: Weekly Target, Current Year, and Previous Year"
        )
        ax.set_xlabel("Total Bookings")
        ax.set_yticks([0, 1, 2], labels=["Last Year", "Actual Year", "Weekly Target"])
        ax.legend()

        # Display the chart in Streamlit
        st.pyplot(fig)

        # Display the DataFrame (optional)
        st.write("Actual Year Data:")
        st.write(data_current_year)
        st.write("Last Year Data:")
        st.write(data_previous_year)


# ================================================================SEC

import datetime
from datetime import timedelta


def create_query_sec(start_date, end_date, previous_year=False):
    query_parts = []

    # Adjust for previous year if required
    if previous_year:
        start_date = start_date.replace(year=start_date.year - 1)
        end_date = end_date.replace(year=end_date.year - 1)

    current_date = start_date
    query_count = []  # Create a list to store COUNT statements

    while current_date <= end_date:
        week_start = current_date
        week_end = week_start + timedelta(days=6)
        if week_end > end_date:
            week_end = end_date

        # Format week label as "MMM DD - MMM DD"
        week_label = f"{week_start.strftime('%b %d')} - {week_end.strftime('%b %d')}"

        query_count.append(
            f"COUNT(CASE WHEN bb.created_at BETWEEN '{week_start}' AND '{week_end}' THEN 1 END) AS \"{week_label}\""
        )

        current_date = week_start + timedelta(days=7)  # Move to the next week

    # Join the COUNT statements with commas
    query_parts.append(", ".join(query_count))

    # Completing the SQL query
    query_parts.insert(0, "SELECT")
    query_parts.append(
        """
        FROM booking_booking bb
        WHERE bb.payment_status = 'CAPTURED'
        AND bb.created_at BETWEEN '{start_date}' AND '{end_date}'
        """.format(
            start_date=start_date, end_date=end_date
        )
    )

    return " ".join(query_parts)


# Main function for analyzing bookings
def analyze_booking_view_sec():
    st.title("Weekly Booking Analysis")

    # Allow user to select a year
    analysis_year = st.selectbox("Select the year for analysis", [2023, 2022, 2021])

    # Calculate dates for the selected year
    jan_1 = datetime.date(analysis_year, 1, 1)
    dec_31 = datetime.date(analysis_year, 12, 31)

    # Set default dates for the date input
    today = datetime.date.today()
    default_start_date = today if jan_1 <= today <= dec_31 else jan_1
    default_end_date = default_start_date + datetime.timedelta(days=6)

    # User selects dates
    vacation_dates = st.date_input(
        "Select your dates",
        (default_start_date, default_end_date),
        min_value=jan_1,
        max_value=dec_31,
        format="MM/DD/YYYY",
    )

    # Check if the returned value is a tuple with two dates
    if not isinstance(vacation_dates, tuple) or len(vacation_dates) != 2:
        st.warning("Please complete the date range.")
        return

    start_date, end_date = vacation_dates

    # Add input fields for editing weekly targets
    st.subheader("Edit Weekly Targets")
    max_weeks = 5  # Maximum of 5 weeks
    weekly_targets_input = [1166] * max_weeks  # Initialize with default values

    for i in range(max_weeks):
        weekly_targets_input[i] = st.number_input(
            f"Week {i+1} Target", value=weekly_targets_input[i]
        )

    if st.button("Analyze Bookings"):
        # Database connection
        conn = get_connection()

        # Queries for current and previous year
        query_current_year = create_query_sec(start_date, end_date)
        query_previous_year = create_query_sec(start_date, end_date, previous_year=True)

        # Retrieve data for current and previous year
        data_current_year = pd.read_sql(query_current_year, conn)
        data_previous_year = pd.read_sql(query_previous_year, conn)

        # Get the dynamic column names for the current and previous year queries
        columns_current_year = [
            col for col in data_current_year.columns if " - " in col
        ]
        columns_previous_year = [
            col for col in data_previous_year.columns if " - " in col
        ]

        # Use only the number of weeks as per the max_weeks variable
        max_length = min(len(columns_current_year), max_weeks)

        # Create a DataFrame for weekly targets
        weekly_targets_df = pd.DataFrame({"Targets": weekly_targets_input[:max_length]})

        # Define a set of distinct colors for the weeks
        week_colors = ["red", "green", "blue", "purple", "orange"]
        # Plotting
        # Increase the figure width to accommodate the legend
        # Increase the figure width to accommodate the legend
        fig, ax = plt.subplots(figsize=(14, 6))  # Increased width

        # Plotting function for each data set
        def plot_data(data, ypos, week_labels):
            cumulative = 0
            for i, week_label in enumerate(week_labels):
                week_val = data[week_label].iloc[0] if week_label in data.columns else 0
                bar = ax.barh(
                    ypos,
                    week_val,
                    left=cumulative,
                    color=week_colors[i % len(week_colors)],
                    label=week_label
                    if ypos == 1
                    else "",  # Label with week range for the legend
                )
                cumulative += week_val

                # Adding data value text on each bar segment
                if week_val > 0:
                    text_x_position = cumulative - (
                        week_val / 2
                    )  # Center of the bar segment
                    ax.text(
                        text_x_position, ypos, str(week_val), va="center", color="white"
                    )

        # Plotting data for current year, previous year, and weekly target
        plot_data(data_current_year, 1, columns_current_year)
        plot_data(data_previous_year, 0, columns_current_year)

        # Adding the weekly target data
        cumulative_target = 0
        for i, week_label in enumerate(columns_current_year):
            target_val = weekly_targets_input[i] if i < len(weekly_targets_input) else 0
            bar = ax.barh(
                2,
                target_val,
                left=cumulative_target,
                color=week_colors[i % len(week_colors)],
            )
            cumulative_target += target_val

            # Adding target value text on each bar segment
            if target_val > 0:
                text_x_position = cumulative_target - (target_val / 2)
                ax.text(text_x_position, 2, str(target_val), va="center", color="white")

        # Set the chart title, labels, and legend
        ax.set_title("Weekly Booking Analysis")
        ax.set_xlabel("Bookings")
        ax.set_yticks([0, 1, 2], labels=["Last Year", "Current Year", "Weekly Target"])

        # Place the legend outside the plot
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), borderaxespad=0.0)

        # Adjust layout to ensure the legend and labels are not cut off
        plt.tight_layout()
        plt.show()

        # Display the chart in Streamlit
        st.pyplot(fig)

        # Transpose the weekly_targets_df DataFrame and use the column names from data_current_year
        weekly_targets_transposed = weekly_targets_df.transpose()
        weekly_targets_transposed.columns = columns_current_year[
            : len(weekly_targets_df)
        ]

        # Compute the backlog, ignoring weeks where current bookings are zero
        backlog_per_week = [
            max(0, target - current)
            for target, current in zip(weekly_targets_input, data_current_year.iloc[0])
            if current > 0
        ]
        total_backlog = sum(backlog_per_week)

        # Display the backlog in Streamlit
        # Display the backlog in red using st.markdown
        st.markdown(
            f"<span style='color: red'>Total Backlog: {total_backlog}</span>",
            unsafe_allow_html=True,
        )

        # st.write("Total Backlog:", total_backlog)
        # Display the transposed weekly targets with the same column names as current year data
        st.write("Weekly Target Data:")
        st.write(weekly_targets_transposed)
        st.write("Current Year Data:")
        st.write(data_current_year)
        st.write("Last Year Data:")
        st.write(data_previous_year)


# ============================================================================= NEW Trend View =================================================================


# def run_query(start_date, end_date, country):
#     query = """
#     SELECT
#         pp.name AS provider_name,
#         COUNT(CASE WHEN bb.created_at BETWEEN %s AND %s THEN 1 END) AS "Previous Week",
#         COUNT(CASE WHEN bb.created_at BETWEEN %s AND %s THEN 1 END) AS "Current Week",
#         CASE
#             WHEN COUNT(CASE WHEN bb.created_at BETWEEN %s AND %s THEN 1 END) > COUNT(CASE WHEN bb.created_at BETWEEN %s AND %s THEN 1 END) THEN 'Trend Up'
#             WHEN COUNT(CASE WHEN bb.created_at BETWEEN %s AND %s THEN 1 END) < COUNT(CASE WHEN bb.created_at BETWEEN %s AND %s THEN 1 END) THEN 'Trend Down'
#             ELSE 'Same'
#         END AS Trend
#     FROM
#         booking_booking bb
#     JOIN
#         experiences_experience ee ON bb.experience_id = ee.id
#     JOIN
#         provider_provider pp ON ee.provider_id = pp.id
#     JOIN
#         core_country co ON cc.country_id = co.id
#     WHERE
#         bb.payment_status = 'CAPTURED'
#         AND co.name = %s
#         AND bb.created_at BETWEEN %s AND %s
#     GROUP BY
#         pp.name
#     ORDER BY
#         pp.name;
#     """

#     with get_connection() as conn:
#         return pd.read_sql(
#             query,
#             conn,
#             params=[start_date, end_date] * 4 + [country, start_date, end_date],
#         )


def run_query(start_date, end_date, country="United Arab Emirates"):
    # Calculating the start and end dates for the previous week
    prev_week_end = start_date - datetime.timedelta(days=1)
    prev_week_start = prev_week_end - datetime.timedelta(days=6)

    query = """
    SELECT 
        pp.name AS provider_name,
        COUNT(CASE WHEN bb.created_at BETWEEN %(prev_week_start)s AND %(prev_week_end)s THEN 1 END) AS "prev_week_count",
        COUNT(CASE WHEN bb.created_at BETWEEN %(current_week_start)s AND %(current_week_end)s THEN 1 END) AS "current_week_count",
        CASE 
            WHEN COUNT(CASE WHEN bb.created_at BETWEEN %(current_week_start)s AND %(current_week_end)s THEN 1 END) > 
                 COUNT(CASE WHEN bb.created_at BETWEEN %(prev_week_start)s AND %(prev_week_end)s THEN 1 END) 
            THEN 'Trend Up'
            WHEN COUNT(CASE WHEN bb.created_at BETWEEN %(current_week_start)s AND %(current_week_end)s THEN 1 END) < 
                 COUNT(CASE WHEN bb.created_at BETWEEN %(prev_week_start)s AND %(prev_week_end)s THEN 1 END) 
            THEN 'Trend Down'
            ELSE 'Same'
        END AS Trend
    FROM 
        booking_booking bb
    JOIN 
        experiences_experience ee ON bb.experience_id = ee.id
    JOIN 
        provider_provider pp ON ee.provider_id = pp.id
    JOIN 
        core_city cc ON ee.city_id = cc.id
    JOIN 
        core_country co ON cc.country_id = co.id
    WHERE 
        bb.payment_status = 'CAPTURED'
        AND co.name = %(country)s
        AND bb.created_at BETWEEN %(prev_week_start)s AND %(current_week_end)s
    GROUP BY 
        pp.name
    ORDER BY 
        pp.name;
    """

    params = {
        "prev_week_start": prev_week_start.strftime("%Y-%m-%d"),
        "prev_week_end": prev_week_end.strftime("%Y-%m-%d"),
        "current_week_start": start_date.strftime("%Y-%m-%d"),
        "current_week_end": end_date.strftime("%Y-%m-%d"),
        "country": country,
    }

    engine = get_connection()
    df = pd.read_sql(query, engine, params=params)
    return df


def trend_indicator_view_sec():
    st.title("Booking Trends Analysis by Country")

    # Country selection (assuming you want to allow different countries to be selected)
    country = st.selectbox(
        "Select Country", ["United Arab Emirates"]
    )  # Modify as needed

    # Determine today's date and next week's Monday
    today = datetime.datetime.now().date()
    next_monday = today + datetime.timedelta((0 - today.weekday()) % 7 + 7)

    # Date range picker with default values set to today and next week's Monday
    start_date, end_date = st.date_input("Select the date range", [today, next_monday])

    if start_date and end_date and country:
        # Adjust start_date and end_date to consider the whole days
        start_date = datetime.datetime.combine(start_date, datetime.time.min)
        end_date = datetime.datetime.combine(end_date, datetime.time.max)

        df = run_query(start_date, end_date, country)

        # Format the date ranges for column naming
        prev_week_label = f"{start_date.strftime('%b %d')} - {(end_date - datetime.timedelta(days=7)).strftime('%b %d')} (Previous Week)"
        current_week_label = f"{(end_date - datetime.timedelta(days=6)).strftime('%b %d')} - {end_date.strftime('%b %d')} (Current Week)"

        # Rename the DataFrame columns
        df.rename(
            columns={
                "prev_week_count": prev_week_label,
                "current_week_count": current_week_label,
            },
            inplace=True,
        )

        # Splitting the DataFrame based on the Trend
        if "trend" in df.columns:
            df_trend_up = df[df["trend"] == "Trend Up"]
            df_trend_down = df[df["trend"] == "Trend Down"]
            df_trend_same = df[df["trend"] == "Same"]

            # Display each DataFrame under a corresponding header
            st.header("Trend Up Results")
            st.dataframe(df_trend_up)

            st.header("Trend Down Results")
            st.dataframe(df_trend_down)

            st.header("Same Results")
            st.dataframe(df_trend_same)
        else:
            st.error("Error: 'Trend' column not found in the DataFrame.")


# ============================================================================= SIDE BAR MENU =================================================================

page_names_to_funcs = {
    "Trend View ": trend_indicator_view_sec,
    "Analyze Booking View ": analyze_booking_view_sec,
    # "Analyze Booking View Ver2": analyze_booking_view_sec,
    "Experience View": experience_view,
    "Keyword View": keyword_view,
    # S"Test View ": test_view,
}

demo_name = st.sidebar.selectbox("Choose a dashboard", page_names_to_funcs.keys())
st.sidebar.write("To view the other pages scroll the content down ➡️")
page_names_to_funcs[demo_name]()
