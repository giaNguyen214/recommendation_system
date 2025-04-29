import pandas as pd
df = pd.read_csv("models/Product.csv")
df.info()

search_result = df[ df['id'] == 192135155]
print("Seach result: ", search_result)
print(df.shape)