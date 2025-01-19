import pandas as pd
import streamlit as st
import os
from datetime import datetime
import logging
import yaml
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import csv
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import io
from PIL import Image as PILImage
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlantInventoryCalculator:
    def __init__(self):
        # Initialize paths
        self.data_dir = Path("Data Files")
        self.pir_dir = self.data_dir / "PIR"
        self.monthly_dir = self.data_dir / "Monthly"
        self.master_dir = self.data_dir / "Master"
        self.calc_dir = self.data_dir / "Calculations"
        
        # Initialize transaction table path
        self.transaction_table = self.master_dir / "TransactionTable.csv"
        
        self.error_number = 0
        self.error_messages = []
        self.config = self.load_config()
        
        # Initialize session state for current table
        if 'current_table' not in st.session_state:
            st.session_state.current_table = str(self.master_dir / "TransactionTable.csv")
        if 'last_total_records' not in st.session_state:
            st.session_state.last_total_records = 0
        
    def load_config(self):
        """Load configuration from yaml file"""
        try:
            with open('config.yaml', 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            self.write_error_log(f"Error loading config: {str(e)}")
            return None
            
    def write_error_log(self, msg):
        """Write error message to log file and store for display"""
        try:
            error_log_path = os.path.join(os.getcwd(), "calculationLog.txt")
            
            with open(error_log_path, "a", encoding='utf-8') as error_log:
                self.error_number += 1
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                error_msg = f"{current_time} [{self.error_number}]: {msg}\n"
                error_log.write(error_msg)
                error_log.flush()
                
            self.error_messages.append(msg)
            
        except Exception as e:
            logger.error(f"Failed to write to calculation log: {str(e)}")
            self.error_messages.append(msg)

    def calculate_metrics(self, df):
        """Calculate various inventory metrics"""
        try:
            metrics = {}
            
            # Total Opening Inventory
            opening_numbers = pd.to_numeric(df['OpeningNumber'], errors='coerce')
            metrics['Total Opening Inventory'] = opening_numbers[opening_numbers != "NA"].sum()
            
            # Number of Plants Destroyed
            plants_destroyed = pd.to_numeric(df['NumberPlantsDestroyed'], errors='coerce')
            metrics['Number of Plants Destroyed'] = plants_destroyed.sum()
            
            # Total Plant Waste Weight
            plant_waste = df[df['WasteType'] == 'Plant']['WasteWeight']
            plant_waste = pd.to_numeric(plant_waste[plant_waste != "NA"], errors='coerce')
            metrics['Total Plant Waste Weight'] = plant_waste.sum()
            
            # Total Veg-Leaf Waste Weight
            veg_waste = df[df['WasteType'] == 'Veg-Leaf']['WasteWeight']
            veg_waste = pd.to_numeric(veg_waste[veg_waste != "NA"], errors='coerce')
            metrics['Total Veg-Leaf Waste Weight'] = veg_waste.sum()
            
            # Total Flower-Leaf Waste Weight
            flower_waste = df[df['WasteType'] == 'Flower-Leaf']['WasteWeight']
            flower_waste = pd.to_numeric(flower_waste[flower_waste != "NA"], errors='coerce')
            metrics['Total Flower-Leaf Waste Weight'] = flower_waste.sum()
            
            # Harvest Waste Weight
            harvest_waste = df[df['WasteType'] == 'Harvest Waste']['WasteWeight']
            harvest_waste = pd.to_numeric(harvest_waste[harvest_waste != "NA"], errors='coerce')
            metrics['Harvest Waste Weight'] = harvest_waste.sum()
            
            # Total Closing Inventory
            closing_numbers = pd.to_numeric(df['ClosingNumber'], errors='coerce')
            metrics['Total Closing Inventory'] = closing_numbers[closing_numbers != "NA"].sum()
            
            return metrics
            
        except Exception as e:
            self.write_error_log(f"Error calculating metrics: {str(e)}")
            return None

    def load_transaction_table(self):
        """Load the transaction table CSV file"""
        try:
            if not os.path.exists(st.session_state.current_table):
                st.error(f"Table file not found: {st.session_state.current_table}")
                return None
            
            # Expected columns
            expected_columns = [
                "TransactionDate", "Action", "LotNum", "PlantForm", "Cultivar", 
                "OpeningNumber", "NumberPlantsDestroyed", "ClosingNumber",
                "WasteWeight", "WasteType", "Details", "HarvestWeight", "CureWeight", 
                "BulkWeight", "DriedWeight"
            ]
            
            # Read CSV directly with expected columns
            df = pd.read_csv(
                st.session_state.current_table,
                names=expected_columns,  # Use expected column names
                header=0,  # Skip the first row (old header)
                engine='python',
                on_bad_lines='skip'
            )
            
            # Fill any missing values with "NA"
            df = df.fillna("NA")
            
            # Ensure all expected columns exist
            for col in expected_columns:
                if col not in df.columns:
                    df[col] = "NA"
            
            # Reorder columns to match expected order
            df = df[expected_columns]
            
            # Store the total number of records
            st.session_state.last_total_records = len(df)
            
            return df
            
        except Exception as e:
            error_msg = f"Error loading TransactionTable.csv: {str(e)}"
            st.error(error_msg)
            self.write_error_log(error_msg)
            return None

    def save_calculations(self, metrics, filtered_df):
        """Save calculation results to a CSV file"""
        try:
            # Create a timestamp for the filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create a DataFrame from the metrics
            metrics_df = pd.DataFrame([metrics])
            
            # Get unique cultivars and plant forms from filtered data
            cultivars = ', '.join(filtered_df['Cultivar'].unique())
            plant_forms = ', '.join(filtered_df['PlantForm'].unique())
            
            # Add filter information
            metrics_df['Date_Range'] = f"{filtered_df['TransactionDate'].min().date()} to {filtered_df['TransactionDate'].max().date()}"
            metrics_df['Cultivars'] = cultivars
            metrics_df['Plant_Forms'] = plant_forms
            
            # Save to Calculations directory
            filename = self.calc_dir / f"Calculations_{timestamp}.csv"
            metrics_df.to_csv(filename, index=False)
            return filename
            
        except Exception as e:
            self.write_error_log(f"Error saving calculations: {str(e)}")
            return None

    def send_email(self, filename, metrics_df, recipient=None):
        """Send calculation results via email"""
        try:
            if not self.config or 'email' not in self.config:
                raise ValueError("Email configuration not found")
                
            email_config = self.config['email']
            
            # Use default recipient if none provided
            if not recipient:
                recipient = email_config.get('default_recipient')
                if not recipient:
                    raise ValueError("No recipient email provided")
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = email_config['sender_email']
            msg['To'] = recipient
            msg['Subject'] = f"Plant Inventory Calculations - {datetime.now().strftime('%Y-%m-%d')}"
            
            # Add metrics as email body
            body = "Plant Inventory Calculations Summary:\n\n"
            for col in metrics_df.columns:
                body += f"{col}: {metrics_df[col].iloc[0]}\n"
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach CSV file
            with open(filename, 'rb') as f:
                attachment = MIMEApplication(f.read(), _subtype='csv')
                attachment.add_header('Content-Disposition', 'attachment', filename=filename)
                msg.attach(attachment)
            
            # Send email
            with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
                server.starttls()
                server.login(email_config['sender_email'], email_config['sender_password'])
                server.send_message(msg)
                
            return True
            
        except Exception as e:
            self.write_error_log(f"Error sending email: {str(e)}")
            return False

    def save_and_email_calculations(self, metrics, filtered_df):
        """Save calculations and optionally email them"""
        filename = self.save_calculations(metrics, filtered_df)
        if not filename:
            return False, None
            
        # Create metrics DataFrame for email
        metrics_df = pd.DataFrame([metrics])
        metrics_df['Date_Range'] = f"{filtered_df['TransactionDate'].min().date()} to {filtered_df['TransactionDate'].max().date()}"
        metrics_df['Cultivars'] = ', '.join(filtered_df['Cultivar'].unique())
        metrics_df['Plant_Forms'] = ', '.join(filtered_df['PlantForm'].unique())
        
        return True, (filename, metrics_df)

    def create_pie_charts(self, filtered_df, metrics):
        """Create pie charts for waste types and other metrics"""
        try:
            # Create subplots for pie charts
            st.subheader("Distribution Charts")
            col1, col2 = st.columns(2)
            
            # Define a colorful palette
            colors = [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
            ]
            
            with col1:
                # Waste Type Distribution
                waste_data = filtered_df[filtered_df['WasteWeight'] != "NA"].groupby('WasteType')['WasteWeight'].sum()
                if not waste_data.empty:
                    fig = px.pie(
                        values=waste_data.values,
                        names=waste_data.index,
                        title='Waste Distribution by Type',
                        color_discrete_sequence=colors
                    )
                    fig.update_traces(
                        textposition='inside',
                        textinfo='percent+label',
                        marker=dict(line=dict(color='white', width=2))
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                # Cultivar Distribution
                cultivar_data = filtered_df.groupby('Cultivar')['NumberPlantsDestroyed'].sum()
                if not cultivar_data.empty:
                    fig = px.pie(
                        values=cultivar_data.values,
                        names=cultivar_data.index,
                        title='Plants Destroyed by Cultivar',
                        color_discrete_sequence=colors
                    )
                    fig.update_traces(
                        textposition='inside',
                        textinfo='percent+label',
                        marker=dict(line=dict(color='white', width=2))
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
            # Store figures in session state
            st.session_state.current_pie_charts = [fig]
            
        except Exception as e:
            self.write_error_log(f"Error creating pie charts: {str(e)}")

    def create_time_series(self, filtered_df):
        """Create time series plots with data sampling"""
        try:
            # Sample data if too large
            daily_metrics = filtered_df.groupby('TransactionDate').agg({
                'OpeningNumber': 'sum',
                'ClosingNumber': 'sum',
                'NumberPlantsDestroyed': 'sum',
                'WasteWeight': lambda x: pd.to_numeric(x, errors='coerce').sum()
            }).reset_index()
            
            # Define colors for different metrics
            color_map = {
                'OpeningNumber': '#1f77b4',  # Blue
                'ClosingNumber': '#2ca02c',  # Green
                'NumberPlantsDestroyed': '#d62728',  # Red
                'WasteWeight': '#ff7f0e'  # Orange
            }
            
            st.subheader("Time Series Analysis")
            
            metric_options = {
                'Inventory Levels': ['OpeningNumber', 'ClosingNumber'],
                'Plants Destroyed': ['NumberPlantsDestroyed'],
                'Waste Weight': ['WasteWeight']
            }
            selected_metric = st.selectbox(
                "Select Metric to Plot",
                options=list(metric_options.keys())
            )
            
            # Create time series plot with colors
            fig = go.Figure()
            
            for column in metric_options[selected_metric]:
                fig.add_trace(
                    go.Scatter(
                        x=daily_metrics['TransactionDate'],
                        y=daily_metrics[column],
                        name=column,
                        mode='lines+markers',
                        line=dict(
                            color=color_map[column],
                            width=2
                        ),
                        marker=dict(
                            color=color_map[column],
                            size=8
                        )
                    )
                )
            
            fig.update_layout(
                title=f'{selected_metric} Over Time',
                xaxis_title='Date',
                yaxis_title='Value',
                hovermode='x unified',
                plot_bgcolor='white',
                paper_bgcolor='white',
                showlegend=True,
                legend=dict(
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='rgba(0,0,0,0.2)',
                    borderwidth=1
                )
            )
            
            fig.update_xaxes(
                gridcolor='rgba(128,128,128,0.2)',
                showline=True,
                linewidth=1,
                linecolor='black',
                mirror=True
            )
            
            fig.update_yaxes(
                gridcolor='rgba(128,128,128,0.2)',
                showline=True,
                linewidth=1,
                linecolor='black',
                mirror=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add moving averages option with colors
            if st.checkbox("Show Moving Averages"):
                window_size = st.slider("Select Window Size (days)", 2, 30, 7)
                
                fig = go.Figure()
                
                for column in metric_options[selected_metric]:
                    base_color = color_map[column]
                    
                    # Original data (lighter shade)
                    fig.add_trace(
                        go.Scatter(
                            x=daily_metrics['TransactionDate'],
                            y=daily_metrics[column],
                            name=f'{column} (Raw)',
                            mode='lines+markers',
                            line=dict(
                                color=base_color,
                                width=1,
                                dash='dot'
                            ),
                            marker=dict(
                                color=base_color,
                                size=6
                            ),
                            opacity=0.5
                        )
                    )
                    
                    # Moving average (darker shade)
                    ma = daily_metrics[column].rolling(window=window_size).mean()
                    fig.add_trace(
                        go.Scatter(
                            x=daily_metrics['TransactionDate'],
                            y=ma,
                            name=f'{column} ({window_size}-day MA)',
                            line=dict(
                                color=base_color,
                                width=3
                            )
                        )
                    )
                
                fig.update_layout(
                    title=f'{selected_metric} Over Time with {window_size}-day Moving Average',
                    xaxis_title='Date',
                    yaxis_title='Value',
                    hovermode='x unified',
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    showlegend=True,
                    legend=dict(
                        bgcolor='rgba(255,255,255,0.8)',
                        bordercolor='rgba(0,0,0,0.2)',
                        borderwidth=1
                    )
                )
                
                fig.update_xaxes(
                    gridcolor='rgba(128,128,128,0.2)',
                    showline=True,
                    linewidth=1,
                    linecolor='black',
                    mirror=True
                )
                
                fig.update_yaxes(
                    gridcolor='rgba(128,128,128,0.2)',
                    showline=True,
                    linewidth=1,
                    linecolor='black',
                    mirror=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            # Store figure in session state
            st.session_state.current_time_series = fig
            
        except Exception as e:
            self.write_error_log(f"Error creating time series plots: {str(e)}")

    def initialize_transaction_table(self):
        """Initialize or get path to transaction table"""
        try:
            filepath = self.master_dir / "TransactionTable.csv"
            
            # Create file with headers if it doesn't exist
            if not filepath.exists():
                headers = [
                    "TransactionDate", "Action", "LotNum", "PlantForm", "Cultivar", 
                    "OpeningNumber", "NumberPlantsDestroyed", "ClosingNumber",
                    "WasteWeight", "WasteType", "HarvestWeight", "CureWeight", 
                    "BulkWeight", "DriedWeight", "Details"
                ]
                filepath.parent.mkdir(parents=True, exist_ok=True)
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
            
            return filepath
            
        except Exception as e:
            self.write_error_log(f"Error initializing transaction table: {str(e)}")
            raise

    def save_to_transaction_table(self, edited_df, filepath):
        """Save to transaction table while preventing duplicates"""
        try:
            # Read existing data if file exists and has content
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                existing_df = pd.read_csv(filepath)
                
                # Create composite key for comparison
                def create_key(df):
                    return df.apply(lambda row: f"{row['TransactionDate']}_{row['Action']}_{row['LotNum']}", axis=1)
                
                existing_keys = create_key(existing_df)
                new_keys = create_key(edited_df)
                
                # Find unique entries
                unique_mask = ~new_keys.isin(existing_keys)
                unique_entries = edited_df[unique_mask]
                
                # Log duplicate entries that were skipped
                duplicates = edited_df[~unique_mask]
                if not duplicates.empty:
                    self.write_error_log(f"Skipped {len(duplicates)} duplicate entries")
                    st.warning(f"Skipped {len(duplicates)} duplicate entries")
                    with st.expander("Show skipped entries"):
                        st.dataframe(duplicates)
                
                # Only append if we have unique entries
                if not unique_entries.empty:
                    unique_entries.to_csv(filepath, mode='a', header=False, index=False)
                    return len(unique_entries)
                return 0
                
            else:
                # If file is empty or doesn't exist, write all entries
                edited_df.to_csv(filepath, index=False)
                return len(edited_df)
                
        except Exception as e:
            raise Exception(f"Error saving to transaction table: {str(e)}")

    def check_for_duplicates(self, edited_df, existing_df):
        """Check for duplicate entries and return duplicates and unique entries"""
        # Create composite key for comparison
        def create_key(df):
            return df.apply(lambda row: f"{row['TransactionDate']}_{row['Action']}_{row['LotNum']}", axis=1)
        
        existing_keys = create_key(existing_df)
        new_keys = create_key(edited_df)
        
        # Find duplicates and unique entries
        duplicates = edited_df[new_keys.isin(existing_keys)]
        unique_entries = edited_df[~new_keys.isin(existing_keys)]
        
        return duplicates, unique_entries

    def save_with_duplicate_check(self, edited_df, filepath):
        """Save data with duplicate checking"""
        try:
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                existing_df = pd.read_csv(filepath)
                duplicates, unique_entries = self.check_for_duplicates(edited_df, existing_df)
                
                if not duplicates.empty:
                    st.warning(f"Found {len(duplicates)} duplicate entries")
                    with st.expander("Show duplicate entries"):
                        st.dataframe(duplicates)
                    
                    # Ask user if they want to proceed with saving unique entries
                    if st.button("Save Unique Entries Only"):
                        if not unique_entries.empty:
                            unique_entries.to_csv(filepath, mode='a', header=False, index=False)
                            st.success(f"Saved {len(unique_entries)} new entries")
                        else:
                            st.info("No new entries to save (all were duplicates)")
                else:
                    # No duplicates, save all entries
                    edited_df.to_csv(filepath, mode='a', header=False, index=False)
                    st.success(f"Saved {len(edited_df)} new entries")
            else:
                # New file, save all entries
                edited_df.to_csv(filepath, index=False)
                st.success(f"Created new file with {len(edited_df)} entries")
                
        except Exception as e:
            st.error(f"Error saving data: {str(e)}")
            self.write_error_log(f"Error during save: {str(e)}")

    def export_to_pdf(self, metrics, filtered_df, figures):
        """Export calculations and visualizations to PDF"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = str(self.calc_dir / f"Report_{timestamp}.pdf")
            
            # Create a temporary directory for images
            temp_dir = self.calc_dir / "temp"
            temp_dir.mkdir(exist_ok=True)
            
            # Create PDF document with RGB color space
            doc = SimpleDocTemplate(
                str(filename),
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            elements = []
            styles = getSampleStyleSheet()
            
            # Add title with color
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Title'],
                textColor=colors.HexColor('#1f77b4'),  # Use a nice blue color
                spaceAfter=30
            )
            title = Paragraph(
                f"Plant Inventory Report - {datetime.now().strftime('%Y-%m-%d')}", 
                title_style
            )
            elements.append(title)
            
            # Add metrics table with better styling
            metrics_data = [[k, f"{v:.2f}" if isinstance(v, float) else str(v)] 
                          for k, v in metrics.items()]
            metrics_table = Table([['Metric', 'Value']] + metrics_data)
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),  # Header background
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),  # Header text
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f2f6')),  # Alternating rows
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#666666')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f2f6')])
            ]))
            elements.append(metrics_table)
            elements.append(Spacer(1, 30))
            
            # Add visualizations with high quality settings
            temp_files = []
            for i, fig in enumerate(figures):
                try:
                    # Convert plotly figure to high-quality image
                    img_bytes = fig.to_image(
                        format="png",
                        width=1200,  # Increased resolution
                        height=800,
                        scale=2.0,  # Higher scale factor for better quality
                        engine="kaleido"
                    )
                    img = PILImage.open(io.BytesIO(img_bytes))
                    
                    # Save to temporary file with high quality
                    img_path = temp_dir / f"temp_fig_{timestamp}_{i}.png"
                    temp_files.append(img_path)
                    img.save(
                        str(img_path),
                        "PNG",
                        optimize=False,
                        quality=100
                    )
                    
                    # Add to PDF with proper sizing
                    elements.append(Image(
                        str(img_path),
                        width=500,
                        height=300,
                        kind='proportional'
                    ))
                    elements.append(Spacer(1, 20))
                except Exception as e:
                    self.write_error_log(f"Error processing figure {i}: {str(e)}")
                    continue
            
            # Build PDF with RGB color space
            doc.build(elements)
            
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    temp_file.unlink()
                except Exception as e:
                    self.write_error_log(f"Error removing temp file {temp_file}: {str(e)}")
            
            try:
                temp_dir.rmdir()
            except Exception:
                pass
                
            return filename
            
        except Exception as e:
            st.error(f"Error creating PDF: {str(e)}")
            self.write_error_log(f"Error creating PDF: {str(e)}")
            return None

    def run(self):
        """Main application loop"""
        st.title("Plant Inventory Calculator")
        
        # Create a session state for tracking reloads
        if 'needs_reload' not in st.session_state:
            st.session_state.needs_reload = False
        
        # Load transaction table
        df = self.load_transaction_table()
        if df is None:
            return
            
        # Add filter section
        st.markdown("---")
        
        # Add filter header and clear button in the same row
        filter_col1, filter_col2 = st.columns([3, 1])
        with filter_col1:
            st.subheader("Filters")
        with filter_col2:
            if st.button("🔄 Clear Filters"):
                # Reset all filter-related session state variables
                if 'start_date' in st.session_state:
                    del st.session_state.start_date
                if 'end_date' in st.session_state:
                    del st.session_state.end_date
                if 'selected_cultivars' in st.session_state:
                    del st.session_state.selected_cultivars
                if 'selected_plant_form' in st.session_state:
                    del st.session_state.selected_plant_form
                st.rerun()
        
        # Create filter columns
        col1, col2 = st.columns(2)
        
        # Date range filter with date-only parsing
        with col1:
            try:
                # Convert dates and strip time component
                df['TransactionDate'] = pd.to_datetime(
                    df['TransactionDate'].str.split().str[0],  # Take only the date part
                    format='%Y-%m-%d'
                )
                
                min_date = df['TransactionDate'].min()
                max_date = df['TransactionDate'].max()
                
                start_date = pd.to_datetime(st.date_input(
                    "Start Date",
                    value=st.session_state.get('start_date', min_date.date()),
                    min_value=min_date.date(),
                    max_value=max_date.date(),
                    key='start_date'
                ))
                
                end_date = pd.to_datetime(st.date_input(
                    "End Date",
                    value=st.session_state.get('end_date', max_date.date()),
                    min_value=min_date.date(),
                    max_value=max_date.date(),
                    key='end_date'
                ))
                
            except Exception as e:
                self.write_error_log(f"Error processing dates: {str(e)}")
                st.error(f"Error processing dates: {str(e)}")
                return
        
        with col2:
            # Cultivar multiselect with "All" option
            unique_cultivars = sorted(df['Cultivar'].unique())
            selected_cultivars = st.multiselect(
                "Select Cultivars",
                options=["All"] + list(unique_cultivars),
                default=st.session_state.get('selected_cultivars', ["All"]),
                key='selected_cultivars'
            )
            
            # Update filter logic to handle "All" selection
            if "All" in selected_cultivars:
                selected_cultivars = unique_cultivars
            
            # Plant Form filter
            unique_plant_forms = sorted(df['PlantForm'].unique())
            all_options = ["All"] + list(unique_plant_forms)
            
            # Get current selection or default to "All"
            current_selection = st.session_state.get('selected_plant_form', "All")
            
            # Find the index of the current selection
            try:
                current_index = all_options.index(current_selection)
            except ValueError:
                current_index = 0  # Default to "All" if current selection is invalid
            
            selected_plant_form = st.selectbox(
                "Select Plant Form",
                options=all_options,
                index=current_index,
                key='selected_plant_form'
            )
        
        # Apply filters
        filtered_df = df.copy()
        
        # Date filter using datetime objects
        date_mask = (filtered_df['TransactionDate'] >= start_date) & \
                   (filtered_df['TransactionDate'] <= end_date)
        filtered_df = filtered_df[date_mask]
        
        # Cultivar filter
        if selected_cultivars:
            filtered_df = filtered_df[filtered_df['Cultivar'].isin(selected_cultivars)]
            
        # Plant Form filter
        if selected_plant_form != "All":
            filtered_df = filtered_df[filtered_df['PlantForm'] == selected_plant_form]
        
        # Display editable table
        st.write("Transaction Table Data:")
        
        # Add reload controls
        reload_col1, reload_col2, reload_col3 = st.columns([1, 1, 2])
        with reload_col1:
            if st.button("🔄 Reload Data"):
                st.session_state.needs_reload = True
                st.rerun()
        
        with reload_col2:
            auto_reload = st.checkbox("Auto Reload", value=True)
            
        with reload_col3:
            if auto_reload:
                reload_interval = st.slider("Reload Interval (seconds)", 
                                         min_value=5, 
                                         max_value=300, 
                                         value=30)
                if 'last_reload_time' not in st.session_state:
                    st.session_state.last_reload_time = datetime.now()
                
                # Check if it's time to reload
                time_since_reload = (datetime.now() - st.session_state.last_reload_time).total_seconds()
                if time_since_reload >= reload_interval:
                    st.session_state.needs_reload = True
                    st.session_state.last_reload_time = datetime.now()
                    st.rerun()
                else:
                    # Show countdown
                    time_left = int(reload_interval - time_since_reload)
                    st.write(f"Next reload in {time_left} seconds")
        
        # Add record count information
        total_records = len(filtered_df)
        
        # Get the total records in transaction table
        try:
            current_transaction_records = len(pd.read_csv("TransactionTable.csv"))
            new_records = current_transaction_records - st.session_state.last_total_records
        except:
            new_records = 0
        
        # Create metrics in columns
        count_col1, count_col2 = st.columns(2)
        with count_col1:
            st.metric("New Records", new_records)
        with count_col2:
            st.metric("Total Records", total_records)
            
        # Add a separator
        st.markdown("---")
        
        # Add pagination for large datasets
        ROWS_PER_PAGE = 1000  # Adjust this number based on your needs
        
        # Calculate total pages
        total_pages = len(filtered_df) // ROWS_PER_PAGE + (1 if len(filtered_df) % ROWS_PER_PAGE > 0 else 0)
        
        # Add page selector
        current_page = st.selectbox("Page", range(1, total_pages + 1)) if total_pages > 1 else 1
        
        # Calculate start and end indices for current page
        start_idx = (current_page - 1) * ROWS_PER_PAGE
        end_idx = min(start_idx + ROWS_PER_PAGE, len(filtered_df))
        
        # Display current page of data
        current_page_df = filtered_df.iloc[start_idx:end_idx].copy()
        
        # Display the table with pagination
        edited_df = st.data_editor(
            current_page_df,
            use_container_width=True,
            num_rows="dynamic",
            key=f"data_editor_{current_page}"
        )

        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Save to Transaction Table"):
                try:
                    filepath = self.initialize_transaction_table()
                    self.save_with_duplicate_check(filtered_df, filepath)  # Use full dataset
                except Exception as e:
                    st.error(f"Error saving to CSV: {str(e)}")
                    self.write_error_log(f"Error during save: {str(e)}")
        
        with col2:
            if st.button("Create New Table from Current Data"):
                try:
                    with st.form("new_table_form"):
                        st.write("Create New Transaction Table")
                        
                        # Directory selection
                        default_dir = os.getcwd()
                        new_dir = st.text_input(
                            "Directory Path", 
                            value=default_dir,
                            help="Leave blank to use current directory"
                        )
                        
                        # Filename input
                        timestamp = datetime.now().strftime("%Y%m%d")
                        default_filename = f"TransactionTable_{timestamp}.csv"
                        new_filename = st.text_input(
                            "Filename",
                            value=default_filename,
                            help="Enter filename with .csv extension"
                        )
                        
                        submitted = st.form_submit_button("Create Table")
                        
                        if submitted:
                            try:
                                if not new_dir.strip():
                                    new_dir = default_dir
                                    
                                if not os.path.exists(new_dir):
                                    st.error("Directory does not exist")
                                    return
                                    
                                new_filepath = os.path.join(new_dir, new_filename)
                                
                                if os.path.exists(new_filepath):
                                    st.error("File already exists. Please choose a different name.")
                                    return
                                
                                self.save_with_duplicate_check(filtered_df, new_filepath)
                                
                                # Option to switch to new table
                                if st.button("Switch to New Table"):
                                    st.session_state.current_table = new_filepath
                                    st.rerun()
                                    
                            except Exception as e:
                                st.error(f"Error creating new table: {str(e)}")
                                self.write_error_log(f"Error creating new table: {str(e)}")
                                
                except Exception as e:
                    st.error(f"Error showing create table form: {str(e)}")
                    self.write_error_log(f"Error showing create table form: {str(e)}")
        
        # Check if we need to reload and reset the flag
        if st.session_state.needs_reload:
            st.session_state.needs_reload = False
            st.rerun()
        
        # Calculate and display metrics using filtered data
        st.markdown("---")
        st.subheader("Calculations")
        
        metrics = self.calculate_metrics(filtered_df)
        if metrics:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Opening Inventory", f"{metrics['Total Opening Inventory']:.0f}")
                st.metric("Number of Plants Destroyed", f"{metrics['Number of Plants Destroyed']:.0f}")
                st.metric("Total Plant Waste Weight", f"{metrics['Total Plant Waste Weight']:.2f} kg")
                st.metric("Total Veg-Leaf Waste Weight", f"{metrics['Total Veg-Leaf Waste Weight']:.2f} kg")
                
            with col2:
                st.metric("Total Flower-Leaf Waste Weight", f"{metrics['Total Flower-Leaf Waste Weight']:.2f} kg")
                st.metric("Harvest Waste Weight", f"{metrics['Harvest Waste Weight']:.2f} kg")
                st.metric("Total Closing Inventory", f"{metrics['Total Closing Inventory']:.0f}")
            
            # Add visualization section
            st.markdown("---")
            st.header("Visualizations")
            
            # Create visualizations
            self.create_pie_charts(filtered_df, metrics)
            self.create_time_series(filtered_df)
            
            # Add save and export buttons
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Save Calculations"):
                    success, result = self.save_and_email_calculations(metrics, filtered_df)
                    if success:
                        filename, _ = result
                        st.success(f"Calculations saved to {filename}")
                    else:
                        st.error("Failed to save calculations")
            
            with col2:
                if st.button("Export to PDF"):
                    # Get current figures
                    figures = []
                    if 'current_pie_charts' in st.session_state:
                        figures.extend(st.session_state.current_pie_charts)
                    if 'current_time_series' in st.session_state:
                        figures.append(st.session_state.current_time_series)
                    
                    pdf_file = self.export_to_pdf(metrics, filtered_df, figures)
                    if pdf_file:
                        st.success(f"Report exported to {pdf_file}")
            
            with col3:
                recipient_email = st.text_input("Email address (optional)")
                if st.button("Save and Email"):
                    success, result = self.save_and_email_calculations(metrics, filtered_df)
                    if success:
                        filename, metrics_df = result
                        if self.send_email(filename, metrics_df, recipient_email):
                            st.success(f"Calculations saved and emailed successfully")
                        else:
                            st.error("Failed to send email")
                    else:
                        st.error("Failed to save calculations")
        
        # Display any errors or warnings
        if self.error_messages:
            st.markdown("---")
            with st.expander("⚠️ Calculation Warnings and Errors", expanded=True):
                for msg in self.error_messages:
                    st.error(msg)
        
        # Add error log viewer button
        if st.button("View Error Log"):
            try:
                os.system('streamlit run "error_log_viewer.py"')
            except Exception as e:
                st.error(f"Error opening Error Log Viewer: {str(e)}")

def main():
    calculator = PlantInventoryCalculator()
    calculator.run()
    
    # Add footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size: 10px; text-align: center;'>Created by Nestech-AI 2025</p>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 