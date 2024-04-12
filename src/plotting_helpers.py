import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# State abbreviation to full name mapping
state_mapping = {
    'FL': 'Florida', 'CA': 'California', 'TX': 'Texas', 'GA': 'Georgia',
    'NY': 'New York', 'IL': 'Illinois', 'PA': 'Pennsylvania', 'NC': 'North Carolina',
    'NJ': 'New Jersey', 'MD': 'Maryland', 'VA': 'Virginia', 'OH': 'Ohio',
    'MI': 'Michigan', 'SC': 'South Carolina', 'AZ': 'Arizona', 'TN': 'Tennessee',
    'NV': 'Nevada', 'LA': 'Louisiana', 'AL': 'Alabama', 'MO': 'Missouri',
    'MA': 'Massachusetts', 'IN': 'Indiana', 'AR': 'Arkansas', 'WA': 'Washington',
    'CO': 'Colorado', 'MS': 'Mississippi', 'CT': 'Connecticut', 'MN': 'Minnesota',
    'WI': 'Wisconsin', 'KY': 'Kentucky', 'UT': 'Utah', 'DE': 'Delaware',
    'OR': 'Oregon', 'OK': 'Oklahoma', 'DC': 'District of Columbia', 'KS': 'Kansas',
    'IA': 'Iowa', 'NM': 'New Mexico', 'NE': 'Nebraska', 'HI': 'Hawaii',
    'RI': 'Rhode Island', 'ID': 'Idaho', 'WV': 'West Virginia', 'NH': 'New Hampshire',
    'ME': 'Maine', 'MT': 'Montana', 'ND': 'North Dakota', 'AK': 'Alaska',
    'SD': 'South Dakota', 'WY': 'Wyoming', 'VT': 'Vermont'
    # Removed territories and minor outlying islands not listed as states
}

# Function to plot top n most common categories
def plot_top_n(df, column, title, n=5, palette_name=None):
    # Generate a color sequence from the seaborn palette
    color_sequence = sns.color_palette(palette_name, n_colors=n).as_hex() if palette_name else None
    
    # Get top n most common values in the specified column
    counts = df[column].value_counts().reset_index()
    counts.columns = [column, 'Count']
    top_n = counts.head(n)
    
    # Create a horizontal bar plot with the seaborn color sequence and remove the legend
    fig = px.bar(top_n, y=column, x='Count', orientation='h', 
                 color=column, color_discrete_sequence=color_sequence)
    fig.update_layout(showlegend=False)
    return fig

# 1. Plotting top 5 most common products
def plot_top_5_products(df_new):
    # df_new = load_process_data(df)
    fig = plot_top_n(df_new, 'Product', 'Top 5 Most Common Products')
    return fig

# 2. Plotting Top 5 common issues
def plot_top_5_issues(df_new):
    # df_new = load_process_data(df)
    fig = plot_top_n(df_new, 'Issue', 'Top 5 Most Common Issues', palette_name='plasma')
    return fig

# 3. Plotting top 5 issues in each product category
def plot_top_5_issues_in_product(df_new):
    # Step 1: Group data by 'Product' and 'Issue', then count occurrences
    grouped_data = df_new.groupby(['Product', 'Issue']).size().reset_index(name='Count')

    # Calculate total issues per product for ordering
    total_issues_per_product = grouped_data.groupby('Product')['Count'].sum().reset_index(name='TotalIssues')

    # Sort products by total issues in descending order
    sorted_products = total_issues_per_product.sort_values('TotalIssues', ascending=False)

    # Step 2: Get top 5 issues for each product sorted by 'Count' in descending order
    top_issues_per_product = (grouped_data.groupby('Product', as_index=False)
                            .apply(lambda x: x.nlargest(5, 'Count'))
                            .reset_index(drop=True))

    # Merge to get the order column (TotalIssues) in top_issues_per_product for sorting
    top_issues_per_product = top_issues_per_product.merge(sorted_products, on='Product')

    # Sort top_issues_per_product DataFrame based on TotalIssues column to ensure the plot respects this order
    top_issues_per_product = top_issues_per_product.sort_values(by=['TotalIssues', 'Count'], ascending=[False, False])

    # Step 3: Create a vertical stacked bar chart
    fig = px.bar(top_issues_per_product, x='Product', y='Count', color='Issue',
                labels={'Count': 'Number of Complaints'}, 
                category_orders={'Product': sorted_products['Product'].tolist()}) # Explicitly set the order of products

    # Update layout to remove legend and adjust dimensions for clarity
    fig.update_layout(showlegend=False, width=900, height=600)
    return fig

# 4.Companies with the Most Complaints in 2023
def plot_top_10_companies_complaints(df_new):
    # Filter data for the year 2023
    df_2023 = df_new[df_new['Date received'].dt.year == 2023]

    # Group data by company name and count the number of complaints for each company
    company_complaint_counts = df_2023['Company'].value_counts()

    top_n = 10
    # Ensure the companies are sorted in ascending order for correct plotting
    top_companies = company_complaint_counts.head(top_n).sort_values(ascending=True)

    # Create a horizontal bar chart using Plotly Express with a nicer color scale
    fig = px.bar(
        x=top_companies.values,
        y=top_companies.index,
        orientation='h',
        color=top_companies.values, # This assigns a color based on the value
        color_continuous_scale=[(0.0, "green"),
                                (0.05, "yellow"),
                                (1.0, "red")], # This is an example of a nice color scale
        labels={'x': 'Number of Complaints', 'y': 'Company'}
    )

    fig.update_layout(
        xaxis=dict(
            title='Number of Complaints',
        ),
        yaxis=dict(
            tickfont=dict(size=10),
        ),
        height=500,
        width=800,
    )

    # To display a color bar, showing the mapping of colors to values
    fig.update_layout(coloraxis_showscale=False)
    return fig

# 5. Top 10 States with the Most Complaints
def plot_top_10_states_most_complaints(df_new):
    # Assuming df_new is your DataFrame and 'State' contains the abbreviations
    # Map state abbreviations to full names
    df_new['State Name'] = df_new['State'].map(state_mapping)

    # Calculate complaint counts by state
    state_complaint_counts = df_new['State Name'].value_counts()

    # Get top 10 states with the most complaint counts
    top_n = 10
    top_states = state_complaint_counts.head(top_n)

    # Create a horizontal bar chart using Plotly Express with a nice color scale
    fig = px.bar(
        x=top_states.values,
        y=top_states.index,
        orientation='h',
        color=top_states.values,  # Assign color based on values
        color_continuous_scale='Turbo',  # A nice color scale
        labels={'x': 'Number of Complaints', 'y': 'State'},
        category_orders={'y': top_states.index.tolist()}
    )

    fig.update_layout(
        yaxis=dict(
            tickfont=dict(size=10),
        ),
        xaxis=dict(
            tickangle=0,
        ),
        height=500,
        width=900,
    )

    # To display a color bar, showing the mapping of colors to values
    fig.update_layout(coloraxis_showscale=False)
    return fig

# 6. Top 10 States with the Least Complaints
def plot_top_10_states_least_complaints(df_new):
    # Map state abbreviations to full names
    df_new['State Name'] = df_new['State'].map(state_mapping)

    # Calculate complaint counts by state
    state_complaint_counts = df_new['State Name'].value_counts()

    # Get top 10 states with the most complaint counts
    top_n = 10
    top_states = state_complaint_counts.tail(top_n)

    # Create a horizontal bar chart using Plotly Express with a nice color scale
    fig = px.bar(
        x=top_states.values,
        y=top_states.index,
        orientation='h',
        color=top_states.values,  # Assign color based on values
        color_continuous_scale='Temps',  # A nice color scale
        labels={'x': 'Number of Complaints', 'y': 'State'},
        category_orders={'x': top_states.index.tolist()}
    )

    fig.update_layout(
        yaxis=dict(
            tickfont=dict(size=10),
        ),
        xaxis=dict(
            tickangle=0,
        ),
        height=500,
        width=900,
    )

    # To display a color bar, showing the mapping of colors to values
    fig.update_layout(coloraxis_showscale=False)

    return fig

# 7. Number of Complaints by Year
def complaints_by_year(df_new):
    monthly_complaints = df_new.copy()
    monthly_complaints = monthly_complaints[monthly_complaints['Date received'].dt.year != 2024]

    monthly_complaints['MonthYear'] = monthly_complaints['Date received'].dt.to_period('M').astype(str)
    monthly_complaints = monthly_complaints.groupby('MonthYear').size().reset_index(name = "NumComplaints")


    fig = px.line(monthly_complaints, x='MonthYear', y='NumComplaints',
                labels={'MonthYear': 'Year', 'NumComplaints': 'Number of Complaints'})

    fig.update_layout(
            width=900,
            height=400
        )
    return fig

# 8. Number of Complaints by State
def complaints_across_states(df_new):
    df_2023 = df_new[df_new['Date received'].dt.year == 2023]

    state_complaints = df_2023.groupby('State').size().reset_index(name='Num_complaints')
    state_complaints['Full_state_name'] = state_complaints['State'].apply(lambda x : state_mapping[x] if x in state_mapping else x)

    fig = px.choropleth(state_complaints,
                        locations='State',
                        locationmode='USA-states',
                        color='Num_complaints',
                        color_continuous_scale='Inferno',
                        scope="usa",
                        hover_name='Full_state_name')
    fig.add_scattergeo(
        locations=state_complaints['State'],    ###codes for states,
        locationmode='USA-states',
        text=state_complaints['State'],
        mode='text',
        hoverinfo='skip',
        textfont=dict(size = 8.5,color='white'))

    fig.update_layout(
        autosize = True,
        geo=dict(
            landcolor='rgb(217, 217, 217)',  
            lakecolor='rgb(255, 255, 255)',  
            bgcolor='rgb(255, 255, 255)' 
        ),
        paper_bgcolor='rgb(255, 255, 255)', 
        margin={"r":0,"t":50,"l":0,"b":0},
        width=1000,
        height=400
    )
    return fig