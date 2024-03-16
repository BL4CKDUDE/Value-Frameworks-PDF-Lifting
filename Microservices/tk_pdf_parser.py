"""
@Author: Tebogo MJ
@Copy DateTime: 15 Friday March 2024 21:59
@Task: Unifi Zindi PDF Lifting Data Science Competition

@Contact: mojela74@gmail.com
"""

import numpy as np
import pandas as pd
import datetime
import time
import re
import os
import sys
import pdfplumber
import tabula

def extract_text_and_tables(pdf_path):
    text_content = ""
    tables = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extract text content from the page
            text_content += page.extract_text()

            # Extract tables from the page
            #page_tables = page.extract_tables(table_settings={"vertical_strategy": "text", "horizontal_strategy": "text"})
            
            #if page_tables:
                #tables.extend(page_tables)

    return text_content, tables

def tabulate(pdf_path, document_id, document_name, group):
    data = []

    # Extract tables from PDF
    tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)

    for table in tables:
        for index, row in table.iterrows():
            activity_metric = row[0] 
            for year, value in row[2:].items(): 
                if not pd.isna(value): 
                    data.append({
                        "document_id": document_id, 
                        "document_name": document_name, 
                        "group": group, 
                        "activity_metrics": {
                            "activity_metric": activity_metric,
                            "year": year,
                            "stat_value": value
                        }
                    })
    return data


    return data