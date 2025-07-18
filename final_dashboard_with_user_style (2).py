import dash 
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
import base64
import io
import zipfile
import tempfile
import os
import re
import random
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

app = dash.Dash(__name__)
app.title = "Drowsiness & Driving Behavior Dashboard"

app.layout = html.Div([
    html.H1("Drowsiness & Driving Behavior Dashboard"),

    html.H2("Upload Files"),
    dcc.Upload(id="upload-csv", children=html.Button("Upload Drowsiness Log CSV")),
    dcc.Upload(id="upload-zip", children=html.Button("Upload Zipped Drowsiness Image Folder")),
    html.Div(id='csv-status'),

    html.Hr(),

    html.Div([
        html.H3("Drowsiness States Over Time"),
        dcc.Graph(id="drowsiness-line"),
    ]),

    html.Div([
        html.H3("Drowsiness State Distribution by User ID"),
        dcc.Graph(id="drowsiness-bar"),
    ]),

    html.Div([
        html.H3("Sample Status Images"),
        html.Div(id='image-columns', style={"display": "flex", "justifyContent": "space-around"})
    ]),

    html.Hr(),

    html.H2("Driving Behavior Analysis"),
    dcc.Upload(id="upload-behavior", children=html.Button("Upload Driving Behavior CSV")),
    html.Div([
        dcc.Graph(id='behavior-bar'),
        html.H4("Preview"),
        html.Div(id="behavior-preview")
    ]),

    html.Hr(),

    html.H2("Upload Driving Sensor Data"),
    dcc.Upload(id="upload-driving-sensor", children=html.Button("Upload Driving Sensor Data CSV")),

    html.H2("Upload Insurance Claim Data"),
    dcc.Upload(id="upload-claim", children=html.Button("Upload Car Insurance Claim CSV")),
    html.Div(id="claim-prediction-output"),

    html.H2("Combined Risk Summary"),
    html.Div([
        html.Div([
            html.H4("📉 Average Risk"),
            html.Div(id="avg-risk", style={"fontSize": "24px", "fontWeight": "bold"})
        ], style={"padding": "10px", "border": "1px solid #ccc", "borderRadius": "10px", "width": "30%"}),

        html.Div([
            html.H4("🔥 High Risk Events"),
            html.Div(id="high-risk-events", style={"fontSize": "24px", "fontWeight": "bold"})
        ], style={"padding": "10px", "border": "1px solid #ccc", "borderRadius": "10px", "width": "30%"}),

        html.Div([
            html.H4("📊 Risk Contribution"),
            dcc.Graph(id="risk-pie")
        ], style={"width": "30%"})
    ], style={"display": "flex", "flexWrap": "wrap", "justifyContent": "space-around"}),

    html.Div([
        html.H4("📈 Risk Trend"),
        dcc.Dropdown(
            id='risk-trend-select',
            options=[
                {"label": "Drowsiness Risk", "value": "Drowsiness Risk"},
                {"label": "Driving Risk", "value": "Driving Risk"},
                {"label": "Total Risk", "value": "Total Risk"}
            ],
            value="Total Risk",
            clearable=False
        ),
        dcc.Graph(id="risk-trend-chart"),
        
    html.H2("Weekly & Monthly Insurance Premium Trends"),

# Frequency selector
dcc.Dropdown(
    id='trend-frequency',
    options=[
        {'label': 'Weekly', 'value': 'W'},
        {'label': 'Monthly', 'value': 'M'}
    ],
    value='W',
    clearable=False,
    style={"width": "200px", "marginBottom": "10px"}
),

# User ID selector (populated dynamically)
dcc.Dropdown(
    id='user-filter',
    multi=True,
    placeholder="Select User ID(s)",
    style={"width": "300px", "marginBottom": "10px"}
),

# Trend graph
dcc.Graph(id="premium-trend-chart"),


        
    ]),

    # html.H4("🚨 Critical Events (e.g., Sleeping + Aggressive)"),
    dash_table.DataTable(id="critical-events-table", style_table={"overflowX": "auto", "display":"none"})
])

global_store = {
    "drowsiness_df": None,
    "yawning_images": [],
    "drowsy_images": [],
    "sleeping_images": []
}

@app.callback(
    Output("csv-status", "children"),
    Output("drowsiness-line", "figure"),
    Output("drowsiness-bar", "figure"),
    Input("upload-csv", "contents"),
    State("upload-csv", "filename")
)
def process_csv(csv_content, filename):
    if csv_content is None:
        return "Please upload a CSV file.", {}, {}

    content_type, content_string = csv_content.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))

    if "Normalized Status" in df.columns:
        df["Status"] = df["Normalized Status"]

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp'])

    df['User ID'] = pd.to_numeric(df['User ID'], errors='coerce')
    df = df.dropna(subset=['User ID'])
    df['User ID'] = df['User ID'].astype(int)

    global_store['drowsiness_df'] = df

    state_counts = df.groupby([df['Timestamp'].dt.date, 'Status']).size().unstack().fillna(0)
    fig1 = px.line(state_counts, markers=True, title='Distribution of Drowsiness States Over Time')
    fig1.update_layout(xaxis_title="Date", yaxis_title="Frequency")

    user_state_counts = df.groupby(['User ID', 'Status']).size().unstack().fillna(0).sort_index()
    fig2 = px.bar(user_state_counts, barmode='stack', title='Drowsiness State Distribution by User ID')
    fig2.update_layout(xaxis_title="User ID", yaxis_title="Frequency")

    return f"{filename} uploaded and processed.", fig1, fig2

@app.callback(
    Output("image-columns", "children"),
    Input("upload-zip", "contents"),
    State("upload-zip", "filename")
)
def process_zip(zip_content, filename):
    if zip_content is None:
        return [html.Div("Please upload a ZIP folder containing drowsiness images.")]

    content_type, content_string = zip_content.split(',')
    decoded = base64.b64decode(content_string)

    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, "temp.zip")
        with open(zip_path, "wb") as f:
            f.write(decoded)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)

        yawning_pattern = re.compile(r'yawning', re.IGNORECASE)
        drowsy_pattern = re.compile(r'drowsy', re.IGNORECASE)
        sleeping_pattern = re.compile(r'sleeping', re.IGNORECASE)

        yawning_images, drowsy_images, sleeping_images = [], [], []

        for root, _, files in os.walk(tmp_dir):
            for file in files:
                filepath = os.path.join(root, file)
                if yawning_pattern.search(file): yawning_images.append(filepath)
                elif drowsy_pattern.search(file): drowsy_images.append(filepath)
                elif sleeping_pattern.search(file): sleeping_images.append(filepath)

        global_store['yawning_images'] = yawning_images
        global_store['drowsy_images'] = drowsy_images
        global_store['sleeping_images'] = sleeping_images

        def img_div(image_list, label):
            if not image_list:
                return html.Div(f"No {label} images found.")
            img = Image.open(random.choice(image_list))
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            encoded = base64.b64encode(buf.getvalue()).decode()
            return html.Img(src=f'data:image/png;base64,{encoded}', style={'height': '250px'})

        return [
            html.Div([html.H5("Yawning"), img_div(yawning_images, "Yawning")]),
            html.Div([html.H5("Drowsy"), img_div(drowsy_images, "Drowsy")]),
            html.Div([html.H5("Sleeping"), img_div(sleeping_images, "Sleeping")]),
        ]

@app.callback(
    Output("behavior-bar", "figure"),
    Output("behavior-preview", "children"),
    Input("upload-behavior", "contents"),
    State("upload-behavior", "filename")
)
def process_behavior(csv_content, filename):
    if csv_content is None or global_store["drowsiness_df"] is None:
        return {}, "Upload a driving behavior file and make sure drowsiness data is available."

    content_type, content_string = csv_content.split(',')
    decoded = base64.b64decode(content_string)
    df_behavior = pd.read_csv(io.StringIO(decoded.decode("utf-8")))

    # Sequential User ID mapping
    df_drowsy = global_store["drowsiness_df"]
    min_len = min(len(df_behavior), len(df_drowsy))

    df_behavior = df_behavior.iloc[:min_len].copy()
    df_behavior["User ID"] = df_drowsy["User ID"].iloc[:min_len].values

    # Behavior label mapping
    class_mapping = {
        1: "Sudden Acceleration",
        2: "Sudden Right Turn",
        3: "Sudden Left Turn",
        4: "Sudden Braking"
    }

    df_behavior["Behavior Label"] = df_behavior["Target"].map(class_mapping)
    df_behavior["Behavior Label"] = df_behavior["Behavior Label"].astype(str)

    # Safe groupby join
    behavior_per_user = df_behavior.groupby("User ID")["Behavior Label"].apply(
        lambda x: ', '.join(str(i) for i in x.dropna().unique())
    ).reset_index()
    behavior_per_user.columns = ["User ID", "Driving Behaviors"]

     # Driving Style Classification
    def classify_driving_style(behaviors):
        aggressive_behaviors = ["Sudden Acceleration", "Sudden Braking", "Sudden Left Turn", "Sudden Right Turn"]
        behavior_list = [b.strip() for b in behaviors.split(",") if b.strip() in aggressive_behaviors]
        count = len(behavior_list)
        if count <= 1:
            return "Normal"
        elif count == 2:
            return "Medium"
        else:
            return "Aggressive"
    behavior_per_user["Driving Style"] = behavior_per_user["Driving Behaviors"].apply(classify_driving_style)    

    global_store["behavior_df"] = df_behavior
    global_store["behavior_summary"] = behavior_per_user

    # Bar chart
    behavior_counts = df_behavior["Behavior Label"].value_counts().reset_index()
    behavior_counts.columns = ["Behavior", "Count"]

    fig = px.bar(behavior_counts, x="Behavior", y="Count",
                 title="Driving Behavior Class Distribution",
                 labels={"Behavior": "Behavior Type", "Count": "Count"})

    preview_table = dash_table.DataTable(
        data=behavior_per_user.head(15).to_dict("records"),
        columns=[{"name": i, "id": i} for i in behavior_per_user.columns],
        style_table={"overflowX": "auto"}
    )

    return fig, preview_table


@app.callback(
    Output("avg-risk", "children"),
    Output("high-risk-events", "children"),
    Output("risk-trend-chart", "figure"),
    Output("risk-pie", "figure"),
    Output("critical-events-table", "data"),
    Output("critical-events-table", "columns"),
    Input("upload-csv", "contents"),
    Input("upload-driving-sensor", "contents"),
    Input("risk-trend-select", "value"),
)
def update_combined_risk(csv_content, sensor_content, selected_trend):
    if csv_content is None or sensor_content is None:
        return "N/A", "N/A", go.Figure(), go.Figure(), [], []

    
    _, csv_string = csv_content.split(',')
    df_drowsy = pd.read_csv(io.StringIO(base64.b64decode(csv_string).decode("utf-8")))
    if "Normalized Status" in df_drowsy.columns:
        df_drowsy["Status"] = df_drowsy["Normalized Status"]
    drowsy_map = {"Active": 0, "Drowsy": 1, "Yawning": 2, "Sleeping": 3}
    df_drowsy["Drowsiness Risk"] = df_drowsy["Status"].map(drowsy_map)
    

    _, sensor_string = sensor_content.split(',')
    df_motion = pd.read_csv(io.StringIO(base64.b64decode(sensor_string).decode("utf-8")))
    behavior_map = {"SLOW": 0, "NORMAL": 1, "AGGRESSIVE": 2}
    df_motion["Driving Risk"] = df_motion["Class"].map(behavior_map)
    

    min_len = min(len(df_drowsy), len(df_motion))
    df_combined = pd.DataFrame({
        "User ID": df_drowsy.loc[:min_len-1, "User ID"].values
        if "User ID" in df_drowsy.columns else [f"User_{i}" for i in range(min_len)],
        "Drowsiness Status": df_drowsy.loc[:min_len-1, "Status"].values,
        "Driving Style": df_motion.loc[:min_len-1, "Class"].values,
        "Drowsiness Risk": df_drowsy.loc[:min_len-1, "Drowsiness Risk"].values,
        "Driving Risk": df_motion.loc[:min_len-1, "Driving Risk"].values
    })

    df_combined["Total Risk"] = (df_combined["Drowsiness Risk"] + df_combined["Driving Risk"]) / 2

     # Summary metrics
    avg_total_risk = df_combined["Total Risk"].mean()
    high_risk_count = (df_combined["Total Risk"] >= 2.5).sum()

     # Trend plot
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(y=df_combined[selected_trend], name=selected_trend))
    fig_line.update_layout(title=f"{selected_trend} Over Time", xaxis_title="Index", yaxis_title="Risk Score (0–3)")


    d_avg = df_combined["Drowsiness Risk"].mean()
    b_avg = df_combined["Driving Risk"].mean()
    fig_pie = px.pie(
        names=["Drowsiness", "Driving"],
        values=[d_avg, b_avg],
        title="Average Contribution to Total Ris"
    )
    critical = df_combined[
        (df_combined["Drowsiness Risk"] >=2) &
        (df_combined["Driving Risk"] >=2)
    ]
    table_data = critical.to_dict("records")
    table_cols = [{"name": i, "id": i} for i in critical.columns]

    return (
        f"{avg_total_risk:.2f}",
        f"{high_risk_count} times",
        fig_line,
        fig_pie,
        table_data,
        table_cols
    )   

@app.callback(
    Output("claim-prediction-output", "children"),
    Input("upload-claim", "contents"),
    State("upload-claim", "filename")
)
def process_claim_file(claim_content, filename):
    if claim_content is None:
        return "Please upload a car insurance claim CSV file."

    try:
        content_type, content_string = claim_content.split(',')
        decoded = base64.b64decode(content_string)
        df_claim = pd.read_csv(io.StringIO(decoded.decode("utf-8")))

        # Assign user ID if missing
        if "User ID" not in df_claim.columns:
            df_claim["User ID"] = list(range(1, len(df_claim) + 1))

        df_claim = df_claim.iloc[:15].copy()

        # Merge Driving Style
        behavior_summary = global_store.get("behavior_summary")
        if behavior_summary is not None:
            df_claim = pd.merge(df_claim, behavior_summary[["User ID", "Driving Style"]], on="User ID", how="left")
        else:
            df_claim["Driving Style"] = "Normal"

        # Merge Drowsiness Risk
        drowsiness_df = global_store.get("drowsiness_df")
        if drowsiness_df is not None and "User ID" in drowsiness_df.columns:
            risk_map = {"Active": 0, "Drowsy": 1, "Yawning": 2, "Sleeping": 3}
            drowsiness_df["Risk Score"] = drowsiness_df["Status"].map(risk_map)
            avg_risk = drowsiness_df.groupby("User ID")["Risk Score"].mean().reset_index()
            df_claim = pd.merge(df_claim, avg_risk, on="User ID", how="left")
            df_claim["Drowsiness Risk"] = df_claim["Risk Score"].fillna(0)
        else:
            df_claim["Drowsiness Risk"] = 0

        # Driving risk mapping
        driving_risk_map = {"Normal": 0, "Medium": 1, "Aggressive": 2}
        df_claim["Driving Risk"] = df_claim["Driving Style"].map(driving_risk_map).fillna(0)

        # Calculate combined risk
        df_claim["Risk_p"] = (df_claim["Driving Risk"] + df_claim["Drowsiness Risk"]) / 2

        # === Detect ANNUAL_MILEAGE Column ===
        mile_col = None
        for col in df_claim.columns:
            if col.strip().lower().replace(" ", "_") in ["annual_mile", "mileage", "annual_mileage", "annual-mileage"]:
                mile_col = col
                break

        if mile_col:
            df_claim["annual_mile"] = pd.to_numeric(df_claim[mile_col], errors='coerce').fillna(10000)
        else:
            df_claim["annual_mile"] = 10000

        # === Premium Calculation ===
        C0 = 300
        wp = 0.05
        df_claim["Estimated Premium"] = df_claim.apply(
            lambda row: round(C0 + row["Risk_p"] * wp * row["annual_mile"]), axis=1
        )

        # === Bar Chart of Estimated Premiums ===
        fig = px.bar(
            df_claim,
            x="User ID",
            y="Estimated Premium",
            title="Estimated Insurance Premium per User",
            text="Estimated Premium"
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis_title="Premium", xaxis_title="User ID", height=400)

        # === Table View ===
        table = dash_table.DataTable(
            data=df_claim[[
                "User ID", "Driving Style", "Driving Risk", "Drowsiness Risk",
                "Risk_p", "annual_mile", "Estimated Premium"
            ]].to_dict("records"),
            columns=[{"name": col, "id": col} for col in [
                "User ID", "Driving Style", "Driving Risk", "Drowsiness Risk",
                "Risk_p", "annual_mile", "Estimated Premium"
            ]],
            style_table={"overflowX": "auto"}    
        )

        return html.Div([
            html.H4(f"{filename} processed using C₀ + Riskₚ × wₚ × annual_mile."),
            table,
            dcc.Graph(figure=fig)
        ])
    
    except Exception as e:
        return f"Error processing file: {str(e)}",

@app.callback(
    Output("user-filter", "options"),
    Output("user-filter", "value"),
    Input("csv-status", "children")
)
def populate_user_filter(_):
    df = global_store.get("drowsiness_df")
    if df is None or "User ID" not in df.columns:
        return [], []

    users = sorted(df["User ID"].unique())
    options = [{"label": f"User {uid}", "value": uid} for uid in users]
    return options, users  # default: select all


@app.callback(
    Output("premium-trend-chart", "figure"),
    Input("trend-frequency", "value"),
    Input("csv-status", "children"),
    Input("user-filter", "value")
)
def generate_premium_trend(frequency, csv_status, selected_users):
    df = global_store.get('drowsiness_df')
    if df is None or 'Timestamp' not in df.columns:
        return go.Figure().update_layout(title="Upload drowsiness log first")

    try:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.dropna(subset=["Timestamp", "User ID"])
        df["User ID"] = df["User ID"].astype(int)

        if "Normalized Status" in df.columns:
            df["Status"] = df["Normalized Status"]

        risk_map = {"Active": 0, "Drowsy": 1, "Yawning": 2, "Sleeping": 3}
        df["Risk Score"] = df["Status"].map(risk_map)

        # Filter by selected users
        if selected_users:
            df = df[df["User ID"].isin(selected_users)]

        grouped = (
            df.groupby([pd.Grouper(key='Timestamp', freq=frequency), 'User ID'])['Risk Score']
            .mean().reset_index(name='Avg Risk')
        )

        base_mileage = 10000 / (52 if frequency == "W" else 12)
        C0 = 300
        wp = 0.05
        grouped["Estimated Premium"] = grouped.apply(
            lambda row: round(C0 + (row["Avg Risk"] / 2) * wp * base_mileage), axis=1
        )

        fig = px.line(
            grouped,
            x='Timestamp',
            y='Estimated Premium',
            color='User ID',
            markers=True,
            title=f"{'Weekly' if frequency == 'W' else 'Monthly'} Insurance Premium Trend per User"
        )
        fig.update_layout(yaxis_title="Premium", xaxis_title="Date")

        return fig

    except Exception as e:
        return go.Figure().update_layout(title=f"Error: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)
