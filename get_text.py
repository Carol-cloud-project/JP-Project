import pandas as pd


df_X=pd.read_csv('/home/rliuaj/balance_sheet/df_X')
df_X=df_X.iloc[:,1:]
columns=df_X.columns

# Create an empty DataFrame to store the results
df_text = pd.DataFrame(columns=["indx", "text"])

# Lists to store data temporarily
indx = []
text_all = []

count = 1

# Loop through df_X rows
for ind in df_X.index:
    try:
        # Start with a base text
        text = "As a financial analyst, you are tasked with estimating next year's balance sheet items for a company, based on the current year's performance of the following items:"

        # Add feature values to the text
        for i in columns:
            text += f" {i} is {df_X.loc[ind, i]},"
        text += '.\n'

        # Append data to the lists (for better performance)
        indx.append(ind)
        text_all.append(text)

        # After every 5 iterations, store data in the DataFrame and write to CSV
        if count % 5 == 0:
            df_text = pd.DataFrame({'indx': indx, 'text': text_all})
            df_text.to_csv("text_all.csv", index=False)

        count += 1
        print(f"Processed {count} rows")

    except Exception as e:
        print(f"Error processing row {ind}: {e}")
        continue

# Final save (in case there are fewer than 5 iterations left)
if len(indx) > 0:
    df_text = pd.DataFrame({'indx': indx, 'text': text_all})
    df_text.to_csv("text_all.csv", index=False)

