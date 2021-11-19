

def ranking(df, k):
    df['bargain'] = df['market_price'] - df['price']
    df = df.sort_values(by=['bargain','price'], ascending=[False, True])
    df = df[:k]
    df.to_csv('result.csv')
    print(df)

