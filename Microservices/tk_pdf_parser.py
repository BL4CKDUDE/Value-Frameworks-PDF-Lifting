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