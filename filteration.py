import re
import pandas as pd

def extract_price_range(query):
    #query = query.content
    query = query.lower().replace(",", "").replace("₹", "").replace("k", "000")

    # Match ranges: 30000 to 50000, 30000-50000, etc.
    range_match = re.search(r"(?:from\s*)?(\d{4,6})\s*(?:to|and|–|-|—)\s*(\d{4,6})", query)
    if range_match:
        low = int(range_match.group(1))
        high = int(range_match.group(2))
        return low, high

    # Match "above 30000", "greater than 40000", etc.
    above_match = re.search(r"(?:above|greater than|more than|over)\s*(\d{4,6})", query)
    if above_match:
        return int(above_match.group(1)), None

    # Match "below 50000", "less than 45000", etc.
    below_match = re.search(r"(?:below|under|less than)\s*(\d{4,6})", query)
    if below_match:
        return None, int(below_match.group(1))

    return None, None


def filter_by_price(df, query):
    low, high = extract_price_range(query)
    df["Price"] = df["Price"].astype(float)

    if low is not None and high is not None:
        return df[(df["Price"] >= low) & (df["Price"] <= high)]
    elif low is not None:
        return df[df["Price"] > low]
    elif high is not None:
        return df[df["Price"] < high]

    return df

def filter_by_keyword(df, query):
    import re

    query = query.lower()

    # Extract keywords (only alphabets)
    words = re.findall(r'\b[a-zA-Z]+\b', query)

    # Filter: Keep rows where any word appears in Brand or Product Name
    mask = df.apply(
        lambda row: any(
            word in row['Brand'].lower() or word in row['Product Name'].lower()
            for word in words
        ), axis=1
    )
    return df[mask]


def filter_by_specifications(df, query):
    query = query.lower()

    # Convert necessary columns to string
    df["RAM"] = df["RAM"].astype(str)
    df["Storage"] = df["Storage"].astype(str)
    df["Processor"] = df["Processor"].astype(str)

    if "16 gb ram" in query or "16gb ram" in query:
        df = df[df["RAM"].str.contains("16", case=False, na=False)]
    if "8 gb ram" in query or "8gb ram" in query:
        df = df[df["RAM"].str.contains("8", case=False, na=False)]
    if "512 gb ssd" in query:
        df = df[df["Storage"].str.contains("512", case=False, na=False)]
    if "1 tb" in query:
        df = df[df["Storage"].str.contains("1", case=False, na=False)]
    if "i7" in query:
        df = df[df["Processor"].str.contains("i7", case=False, na=False)]
    if "i5" in query:
        df = df[df["Processor"].str.contains("i5", case=False, na=False)]
    if "ryzen 7" in query:
        df = df[df["Processor"].str.contains("ryzen 7", case=False, na=False)]
    if "ryzen 5" in query:
        df = df[df["Processor"].str.contains("ryzen 5", case=False, na=False)]

    return df


def filter_by_purpose(df, query):
    query = query.lower()

    # Convert necessary columns to string
    df["Processor"] = df["Processor"].astype(str)
    df["RAM"] = df["RAM"].astype(str)
    df["Specifications"] = df["Specifications"].astype(str)

    if "gaming" in query or "game" in query:
        return df[
            df["Processor"].str.contains("i5|i7|i9|ryzen 5|ryzen 7|ryzen 9", case=False, na=False) &
            df["RAM"].str.contains(r"8\s*gb|16\s*gb|32\s*gb", case=False, na=False) &
            df["Specifications"].str.contains("graphic|nvidia|geforce|gtx|rtx|mx", case=False, na=False)
        ]

    elif "office" in query:
        return df[
            df["Specifications"].str.contains("battery|webcam|office|zoom|teams", case=False, na=False) &
            df["RAM"].str.contains(r"8\s*gb|16\s*gb", case=False, na=False)
        ]

    elif "student" in query:
        df["Price"] = df["Price"].astype(float)
        return df[df["Price"] < 50000]

    elif "remote" in query or "remote work" in query or "work from home" in query:
        return df[df["Specifications"].str.contains("battery|portable|lightweight", case=False, na=False)]

    return df

