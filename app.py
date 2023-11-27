# DASHBOARD FOR THE BOOKING and ACTUAL
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

import requests
import os
import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt

load_dotenv()


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
                        "color": f"rgba(0, 255, 0, 1)",
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


# def create_gauge_with_color_and_data(week_values, total_value=7000):
#     # Calculate cumulative values and percentages
#     #cumulative_values = [sum(week_values[: i + 1]) for i in range(len(week_values))]
#     cumulative_values = [1201, 1024, 894, 913]
#     percentages = [round((value / total_value) * 100, 2) for value in cumulative_values]

#     # Create gauge figure with RGBA color based on percentages
#     fig = go.Figure(
#         go.Indicator(
#             number=dict(font=dict(size=20)),
#             mode="gauge+number",
#             value=cumulative_values[-1],
#             domain={"x": [0, 1], "y": [0, 1]},
#             title={"text": "Actual 4-Week Progress (%)", "font": {"size": 24}},
#             gauge={
#                 "axis": {"range": [None, total_value], "tickcolor": "darkblue"},
#                 "bar": {"color": "darkblue"},
#                 "steps": [
#                     {
#                         "range": [0, cumulative_values[0]],
#                         "color": f"rgba(0, 255, 0, {percentages[0]/100})",
#                     },
#                     {
#                         "range": [cumulative_values[0], cumulative_values[1]],
#                         "color": f"rgba(0, 0, 255, {percentages[1]/100})",
#                     },
#                     {
#                         "range": [cumulative_values[1], cumulative_values[2]],
#                         "color": f"rgba(255, 0, 0, {percentages[2]/100})",
#                     },
#                     {
#                         "range": [cumulative_values[2], cumulative_values[3]],
#                         "color": f"rgba(255, 255, 0, {percentages[3]/100})",
#                     },
#                 ],
#             },
#         )
#     )

#     # Adding annotations for weekly values
#     for i, value in enumerate(week_values):
#         fig.add_annotation(
#             x=0.5,
#             y=0.5 - (0.1 * i),
#             xref="paper",
#             yref="paper",
#             text=f"Week {i+1}: {value} ({percentages[i]}%)",
#             showarrow=False,
#         )

#     # Calculate trend arrow and color based on the last two weeks
#     trend_arrow = "↑" if week_values[-1] >= week_values[-2] else "↓"
#     trend_color = "green" if trend_arrow == "↑" else "red"

#     # Adding trend arrow annotation
#     fig.add_annotation(
#         x=0.5,
#         y=-0.2,
#         xref="paper",
#         yref="paper",
#         text=f"Trend: {trend_arrow}",
#         font={"color": trend_color, "size": 18},
#         showarrow=False,
#     )

#     fig.update_layout(
#         paper_bgcolor="lavender", font={"color": "darkblue", "family": "Arial"}
#     )
#     return fig


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
    import streamlit as st
    import pandas as pd

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

    # Headers using environment variables
    # headers = {
    #     "X-CleverTap-Account-Id": os.getenv("CLEVERTAP_ACCOUNT_ID"),
    #     "X-CleverTap-Passcode": os.getenv("CLEVERTAP_PASSCODE"),
    #     "Content-Type": "application/json",
    # }

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
    # Generate a word cloud using frequencies
    wordcloud = WordCloud(
        width=800, height=400, background_color="white"
    ).generate_from_frequencies(keywords)
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    st.pyplot(plt)


def convert_to_dataframe(keywords):
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(list(keywords.items()), columns=["Keyword", "Count"])
    return df


def to_csv(df):
    # Convert DataFrame to CSV
    return df.to_csv().encode("utf-8")


def keyword_view():
    st.title("CleverTap Keyword Analysis")

    # Date pickers for start and end date
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")

    if st.button("Fetch Keywords"):
        # Convert dates to required format
        from_date = start_date.strftime("%Y%m%d")
        to_date = end_date.strftime("%Y%m%d")

        keywords = fetch_keywords_from_clevertap(from_date, to_date)
        # print(keywords, "keysssz")
        # Remove the key '-1'
        if "-1" in keywords:
            del keywords["-1"]
        if keywords:
            display_wordcloud(keywords)
            df = convert_to_dataframe(keywords)
            st.dataframe(
                df, use_container_width=True
            )  # Display the DataFrame in Streamlit

            # Create a download button for the CSV
            csv = to_csv(df)
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name="keywords.csv",
                mime="text/csv",
            )


# ================================================================ PRODUCT VIEWS =================================================================


@st.cache_data
def fetch_product_views(from_date, to_date):
    EVENT_NAME = "Product Viewed"

    # headers = {
    #     "X-CleverTap-Account-Id": os.getenv("CLEVERTAP_ACCOUNT_ID"),
    #     "X-CleverTap-Passcode": os.getenv("CLEVERTAP_PASSCODE"),
    #     "Content-Type": "application/json",
    # }

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


def display_horizontal_bar_chart(data):
    if data:
        product_data = data.get("eventPropertyProductViews", {}).get("STR", {})
        df = pd.DataFrame(product_data.items(), columns=["Product Name", "Count"])
        df.sort_values(by="Count", ascending=True, inplace=True)
        # Filter out the entry with Product Name "-1"
        df = df[df["Product Name"] != "-1"]
        # Display the DataFrame
        st.dataframe(df, use_container_width=True)
        # Using Plotly to create a horizontal bar chart
        fig = px.bar(
            df,
            y="Product Name",
            x="Count",
            orientation="h",
            color="Count",
            color_continuous_scale="Blues",
            height=800,
        )
        fig.update_layout(
            title_text="Top 26 Product Views",
            xaxis_title="Count",
            yaxis_title="Product Name",
        )
        st.plotly_chart(fig, use_container_width=True)


def experience_view():
    st.title("CleverTap Experience View Analysis")
    # Date pickers for start and end date
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")

    if st.button("Fetch Experience Views"):
        from_date = start_date.strftime("%Y%m%d")
        to_date = end_date.strftime("%Y%m%d")
        data = fetch_product_views(from_date, to_date)
        display_horizontal_bar_chart(data)


page_names_to_funcs = {
    "Experience View": experience_view,
    "Keyword View": keyword_view,
    "Trend View ": trend_indicator_view,
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
st.sidebar.write("To view the other pages scroll the content down ➡️")
page_names_to_funcs[demo_name]()
