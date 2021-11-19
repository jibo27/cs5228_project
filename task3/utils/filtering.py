import pandas as pd

# def m_strip(x):
#     print(x['manufactured'].dtype)
#     return x['manufactured']
    # return int(x['manufactured'].strip())

# def p_strip(x):
#     print(x['price'].dtype)
#     return x['price']

def my_search(cons):
    filename = 'data/task3.csv'
    df = pd.read_csv(filename)
    # df['manufactured'] = df.apply(m_strip,axis=1)
    # df['price'] = df.apply(p_strip,axis=1)
    s_m, e_m = 0,0 
    if cons[1]!= -1:
        s_m,e_m = int(cons[1].split('-')[0]), int(cons[1].split('-')[1])
    s_p, e_p = 0,0
    if cons[3]!= -1:
        s_p, e_p = float(cons[3].split('-')[0]), float(cons[3].split('-')[1])

    b0 = (df['make']==cons[0])
    b2 = (df['transmission']==cons[2])
    # b1 = ((df['manufactured']>= s_m) and (df['manufactured']<= e_m))
    # b3 = ((df['price']>= s_p) and (df['price']<= e_p)).all()
    if cons[0] != -1:
        df = df.loc[b0]
    if cons[1] != -1:
        df = df.loc[(df['manufactured']>= s_m)]
        df = df.loc[(df['manufactured']<= e_m)]
    if cons[2] != -1:
        df = df.loc[b2]
    if cons[3] != -1:
        df = df.loc[df['price']>= s_p]
        df = df.loc[df['price']<= e_p]
     

    return df 
