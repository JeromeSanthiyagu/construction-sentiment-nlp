import pandas as pd
import random
from datetime import datetime, timedelta

def generate_data_with_dates(num_samples=500):
    positive_templates = [
        "Work completed on time.",
        "Great team coordination today.",
        "Task finished ahead of schedule.",
        "Material delivery arrived early.",
        "Safety inspection passed successfully.",
        "Good progress on the foundation work.",
        "Weather was perfect for concreting.",
        "Client is happy with the progress.",
        "Efficient workflow today.",
        "All targets met for the day."
    ]
    
    neutral_templates = [
        "Inspection scheduled for tomorrow.",
        "Meeting with the architect at 2 PM.",
        "Material delivery expected next week.",
        "Routine maintenance check done.",
        "Site office closed for lunch.",
        "Workers are on break.",
        "Pending approval from the structural engineer.",
        "Daily report submitted.",
        "Shift change at 6 PM.",
        "Inventory check in progress."
    ]
    
    negative_templates = [
        "Delay due to shortage of cement.",
        "Heavy rain affected the work progress.",
        "Machinery breakdown caused a halt.",
        "Labor shortage today.",
        "Safety violation reported.",
        "Material quality is poor.",
        "Conflict between workers.",
        "Budget overrun for this phase.",
        "Permit issues causing delays.",
        "Accident on site, work stopped."
    ]
    
    data = []
    
    # Generate dates for the last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    for _ in range(num_samples):
        sentiment = random.choice(['Positive', 'Neutral', 'Negative'])
        if sentiment == 'Positive':
            note = random.choice(positive_templates)
        elif sentiment == 'Neutral':
            note = random.choice(neutral_templates)
        else:
            note = random.choice(negative_templates)
            
        # Add some variation
        if random.random() > 0.8:
            note += " " + random.choice(["Updates to follow.", "Will report tomorrow.", "Noted in log."])
        
        # Random date within the last 30 days
        random_date = start_date + timedelta(days=random.randint(0, 30))
        date_str = random_date.strftime('%Y-%m-%d')
        
        data.append({'Date': date_str, 'Note': note, 'Sentiment': sentiment})
        
    df = pd.DataFrame(data)
    df = df.sort_values('Date')
    df.to_csv('construction_notes.csv', index=False)
    print(f"Generated {num_samples} samples with dates in construction_notes.csv")

if __name__ == "__main__":
    generate_data_with_dates()
