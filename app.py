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

# ================================================================SEC

import datetime
from datetime import timedelta

@st.cache_data
def create_query_sec(start_date, end_date, previous_year=False):
    query_parts = []

    current_date = start_date
    query_count = []  # Create a list to store COUNT statements

    while current_date < end_date:
        week_start = current_date
        week_end = week_start

        # Find the next Tuesday
        while week_end.weekday() != 1:
            week_end += timedelta(days=1)

        # Extend to include the Tuesday
        week_end += timedelta(days=1)

        # If it's the last week of the year, adjust the week to include January 1st of the next year
        if week_end.year > week_start.year:
            week_end = datetime.date(week_end.year, 1, 1)
        

        # Ensure the week does not exceed the end_date
        if week_end > end_date:
            week_end = end_date + timedelta(days=1)  # Include the last day

        # Format week label as "MMM DD - MMM DD"
        week_label = f"{week_start.strftime('%b %d')} - {(week_end - timedelta(days=1)).strftime('%b %d')}"

        # Adjust dates for previous year if required
        adjusted_week_start = week_start.replace(year=week_start.year - 1) if previous_year else week_start
        adjusted_week_end = week_end.replace(year=week_end.year - 1) if previous_year else week_end

        query_count.append(
            f"COUNT(CASE WHEN bb.created_at >= '{adjusted_week_start}' AND bb.created_at < '{adjusted_week_end}' THEN 1 END) AS \"{week_label} Weekly Booking\""
        )

        # Move to the next week
        current_date = week_end

    # Join the COUNT statements with commas
    query_parts.append(", ".join(query_count))

    # Completing the SQL query
    adjusted_start_date = start_date.replace(year=start_date.year - 1) if previous_year else start_date
    adjusted_end_date = end_date.replace(year=end_date.year - 1) if previous_year else end_date

    # st.write(adjusted_week_start, adjusted_week_end)
    query_parts.insert(0, "SELECT")
    query_parts.append(
        f"""
        FROM booking_booking bb
        JOIN experiences_experience ee ON bb.experience_id = ee.id
        JOIN core_city cc ON ee.city_id = cc.id
        JOIN core_country co ON cc.country_id = co.id
        WHERE bb.payment_status = 'CAPTURED'
        AND co.name = 'United Arab Emirates'
        AND bb.created_at >= '{adjusted_start_date}' AND bb.created_at < '{adjusted_week_end}'
        """
    )

    return " ".join(query_parts)

@st.cache_data
def get_bookings_for_today():
    # Get today's date
    today = datetime.date.today()

    # Calculate yesterday's date
    yesterday = today - datetime.timedelta(days=1)

    # Format dates for the SQL query
    start_date = yesterday.strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')

    # SQL query
    query = f"""
    SELECT
        COUNT(*) AS bookings_count
    FROM booking_booking bb
    JOIN experiences_experience ee ON bb.experience_id = ee.id
    JOIN core_city cc ON ee.city_id = cc.id
    JOIN core_country co ON cc.country_id = co.id
    WHERE bb.payment_status = 'CAPTURED'
    AND co.name = 'United Arab Emirates'
    AND bb.created_at >= '{start_date}' AND bb.created_at < '{end_date}'
    """

    # Execute the query and return the result
    conn = get_connection()
    result = pd.read_sql(query, conn)
    return result.iloc[0]['bookings_count']

def is_selected_month_different_from_current(selected_date):
    current_month = datetime.date.today().month
    selected_month = selected_date.month

    # Check if the selected month is different from the current month
    return selected_month != current_month

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
    selected_date = vacation_dates[0]  # this is important to prevent the decrease of backlogs when the selected month is already in the past

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

        # # Print SQL queries used for the analysis
        # st.text("SQL Query for Current Year:")
        # st.write(query_current_year)
        # st.text("SQL Query for Previous Year:")
        # st.write(query_previous_year)

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
        thisdayminus = get_bookings_for_today()
        # st.write(thisdayminus)
        # st.write(backlog_per_week)
        # st.write(is_selected_month_different_from_current(selected_date))
        if is_selected_month_different_from_current(selected_date):
            total_backlog = sum(backlog_per_week)
        else:
            total_backlog = (sum(backlog_per_week)-thisdayminus)

        # Display the backlog in Streamlit
        # Display the backlog in red using st.markdown
        st.markdown(
            f"<span style='color: red'>Total Backlog: -{total_backlog}</span>",
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



def run_query(week1_start, week1_end, week2_start, week2_end, week3_start, week3_end, week4_start, week4_end, country="United Arab Emirates"):
    query = """
    SELECT 
        pp.name AS provider_name,
        COUNT(CASE WHEN bb.created_at BETWEEN %(week1_start)s AND %(week1_end)s THEN 1 END) AS "Week 1 %(week1_start)s - %(week1_end)s",
        COUNT(CASE WHEN bb.created_at BETWEEN %(week2_start)s AND %(week2_end)s THEN 1 END) AS "Week 2 %(week2_start)s - %(week2_end)s",
        COUNT(CASE WHEN bb.created_at BETWEEN %(week3_start)s AND %(week3_end)s THEN 1 END) AS "Week 3 %(week3_start)s - %(week3_end)s",
        COUNT(CASE WHEN bb.created_at BETWEEN %(week4_start)s AND %(week4_end)s THEN 1 END) AS "Week 4 %(week4_start)s - %(week4_end)s",
        CASE
            WHEN COUNT(CASE WHEN bb.created_at BETWEEN %(week2_start)s AND %(week2_end)s THEN 1 END) = 0 AND 
                COUNT(CASE WHEN bb.created_at BETWEEN %(week3_start)s AND %(week3_end)s THEN 1 END) = 0 AND
                COUNT(CASE WHEN bb.created_at BETWEEN %(week4_start)s AND %(week4_end)s THEN 1 END) > 0
            THEN 'New Booking' 
            WHEN COUNT(CASE WHEN bb.created_at BETWEEN %(week4_start)s AND %(week4_end)s THEN 1 END) >= 
                COUNT(CASE WHEN bb.created_at BETWEEN %(week3_start)s AND %(week3_end)s THEN 1 END) + 2
            THEN 'Trend Up'
            WHEN COUNT(CASE WHEN bb.created_at BETWEEN %(week4_start)s AND %(week4_end)s THEN 1 END) <= 
                COUNT(CASE WHEN bb.created_at BETWEEN %(week3_start)s AND %(week3_end)s THEN 1 END) - 2
            THEN 'Trend Down'
            WHEN ABS(COUNT(CASE WHEN bb.created_at BETWEEN %(week4_start)s AND %(week4_end)s THEN 1 END) - 
                    COUNT(CASE WHEN bb.created_at BETWEEN %(week3_start)s AND %(week3_end)s THEN 1 END)) <= 1
            THEN 'Same'
            
            ELSE 'Not Defined'
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
        AND bb.created_at BETWEEN %(week1_start)s AND %(week4_end)s
    GROUP BY 
        pp.name
    ORDER BY 
        pp.name;

    """

    params = {
        "week1_start": week1_start.strftime("%Y-%m-%d"),
        "week1_end": week1_end.strftime("%Y-%m-%d"),
        "week2_start": week2_start.strftime("%Y-%m-%d"),
        "week2_end": week2_end.strftime("%Y-%m-%d"),
        "week3_start": week3_start.strftime("%Y-%m-%d"),
        "week3_end": week3_end.strftime("%Y-%m-%d"),
        "week4_start": week4_start.strftime("%Y-%m-%d"),
        "week4_end": week4_end.strftime("%Y-%m-%d"),
        "country": country,
    }

    engine = get_connection()
    df = pd.read_sql(query, engine, params=params)
    return df

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

def trend_indicator_view_sec():
    st.title("Booking Trends Analysis by Country")

    # Country selection (assuming you want to allow different countries to be selected)
    country = st.selectbox("Select Country", ["United Arab Emirates"])

    # Allow user to input four sets of dates
    week1_start, week1_end = st.date_input("Select Week 1 Date Range", [datetime.date(2023, 10, 24), datetime.date(2023, 10, 31)])
    week2_start, week2_end = st.date_input("Select Week 2 Date Range", [week1_end + datetime.timedelta(days=1), week1_end + datetime.timedelta(days=7)])
    week3_start, week3_end = st.date_input("Select Week 3 Date Range This should include the previous week", [week2_end + datetime.timedelta(days=1), week2_end + datetime.timedelta(days=7)])
    week4_start, week4_end = st.date_input("Select Week 4 Date Range This should include the current week", [week3_end + datetime.timedelta(days=1), week3_end + datetime.timedelta(days=7)])

    # Button to trigger the analysis
    if st.button('Start Analysis'):
        if country and all([week1_start, week1_end, week2_start, week2_end, week3_start, week3_end, week4_start, week4_end]):
            # Call the updated query function
            df = run_query(week1_start, week1_end, week2_start, week2_end, week3_start, week3_end, week4_start, week4_end, country)

            # # Displaying the DataFrame
            st.header("Booking Trends Results")
            # st.dataframe(df)

            # Splitting the DataFrame based on the Trend
            if "trend" in df.columns:
                df_trend_up = df[df["trend"] == "Trend Up"]
                df_trend_down = df[df["trend"] == "Trend Down"]
                df_trend_same = df[df["trend"] == "Same"]
                df_new_booking = df[df["trend"] == "New Booking"]  # New booking category

                # Display each DataFrame under a corresponding header
                st.header("Trend Up Results")
                st.dataframe(df_trend_up)

                st.header("Trend Down Results")
                st.dataframe(df_trend_down)

                st.header("Same Results")
                st.dataframe(df_trend_same)

                st.header("New Booking Results")  # Display new booking results
                st.dataframe(df_new_booking)
            else:
                st.error("Error: 'Trend' column not found in the DataFrame.")
        else:
            st.error("Please select all date ranges and a country to start the analysis.")

        csv = convert_df(df)

        st.download_button(
        "Press to Download the whole booking csv",
        csv,
        "file.csv",
        "text/csv",
        key='download-csv'
        )




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
