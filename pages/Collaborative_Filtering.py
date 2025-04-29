import streamlit as st 
st.set_page_config(layout="wide", page_title="Collaborative filtering")

import pandas as pd 

# st.markdown("""
#     <style>
#     p {
#         font-size: 22px !important;
#     }
#     li {
#         font-size: 22px !important;
#     }
#     </style>
# """, unsafe_allow_html=True)

orin = {
    "Product": ["AS - NÓN LƯỠI TRAI SỐNG ĐẦY", "Kính lão nam nữ không viền", "Túi YUUMY", "Vớ nam cổ thấp Khatoco", "Dây nịt nam, thắt lưng nam da cao cấp"],
    "User 1": [5, 5, -1, 0, 0],
    "User 2": [5, -1, 4, 0, 0],
    "User 3": [0, -1, 0, 5, 5],
    "User 4": [0, 0, -1, 4, -1],
    "Price": [0.9, 1, 0.99, 0.1, 0],
    "Material": [0, 0.01, 0, 1, 0.9]
}

data = pd.DataFrame(orin)
data1 = data.drop(columns=["Price", "Material"])
data2 = data[["Product", "Price", "Material"]]

st.markdown('<h2 style="color:#9370DB;">Overview of the Algorithm</h2>', unsafe_allow_html=True)
st.write("Assume we have a user rating table for products like this (products not yet rated by users are marked as -1)")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.table(data1)
st.write("Some notations")
st.markdown("""
- n<sub>u</sub>: number of users  
- n<sub>p</sub>: number of products  
- r(i, j) = 1 if user j rated product i  
- y<sup>(i, j)</sup>: rating value if r(i, j) == 1. This is the label used for training  
- For consistency, index i is for product, and j is for user  
""", unsafe_allow_html=True)

st.write("Assume for each product, we have its feature vector x. For example, whether the price is good and its level, where 1 is the best and 0 is the worst. The material might be good or bad, the color might be visually appealing, etc.")
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.table(data2)

st.markdown("From that, we define the feature vector x<sup>(i)</sup> for each product, for example:", unsafe_allow_html=True)
st.markdown("""
- AS - NÓN LƯỠI TRAI SỐNG ĐẦY: x<sup>(1)</sup> = [0.9, 0]  
- Kính lão nam nữ không viền: x<sup>(2)</sup> = [1, 0.01]  
""", unsafe_allow_html=True)

st.markdown("We need to build a model with parameters w<sup>(j)</sup> and b<sup>(j)</sup> to predict user rating for a product based on that product’s feature vector: rating ≈ w<sup>(j)</sup>.x<sup>(i)</sup> + b<sup>(j)</sup>. And define the loss function as follows:", unsafe_allow_html=True)

st.write("To predict user j’s ratings for all products:")
col1, col2 = st.columns([1.5, 3])
with col2:
    st.markdown(r"""
$J(w^{(j)}, b^{(j)}) = \frac{1}{2} \sum_{i: r(i, j)==1} (w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i, j)})^2$
""")
st.markdown("""Here, y<sup>(i, j)</sup> is the rating value from available data, used as the label to train and evaluate model accuracy.  
w<sup>(j)</sup>.x<sup>(i)</sup> + b<sup>(j)</sup> is the model prediction.  
The condition i: r(i, j) == 1 is to filter only the products rated by user j (because only those have y<sup>(i, j)</sup> to train the model).""", unsafe_allow_html=True)

st.write("Similarly, to find parameters for all users, we compute the total loss function for all users:")
col1, col2 = st.columns([1.5, 3])
with col2:
    st.markdown(r"""
$J(w^{(1)}...w^{(n_{u})}, b^{(1)}...b^{(n_{u})}) = \sum_{j=1}^{n_{u}} J(w^{(j)}, b^{(j)})$
""")
st.write("What’s left is to minimize the loss function, thereby finding the optimal parameters for the model.")
st.info("Up to this point, the problem is similar to supervised learning.")

st.markdown('<h2 style="color:#9370DB;">What if there’s no feature vector for products?</h2>', unsafe_allow_html=True)
temp = data.copy()
temp["Price"] = [-1] * 5
temp["Material"] = [-1] * 5
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.table(temp)

st.write("Just like before where we assumed x is known and solved for w, b. Now assume w and b are known, and we solve for x.")
col1, col2 = st.columns([1.5, 3])
with col2:
    st.markdown(r"""
$J(x^{(i)}) = \frac{1}{2} \sum_{j: r(i, j)==1} (w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i, j)})^2$
""")
st.write("Similarly, we find features for all products:")
col1, col2 = st.columns([1.5, 3])
with col2:
    st.markdown(r"""
$J(x^{(1)}...x^{(n_{p})}) = \sum_{i=1}^{n_{p}} J(x^{(i)})$
""")

st.markdown('<h2 style="color:#9370DB;">Summary</h2>', unsafe_allow_html=True)
st.markdown(r"Combining the two formulas, the loss function becomes a function of variables $w^{(1)}...w^{(n_{u})}$, $b^{(1)}...b^{(n_{u})}$, and $x^{(1)}...x^{(n_{p})}$", unsafe_allow_html=True)
col1, col2 = st.columns([1.5, 3])
with col2:
    st.markdown(r"""
$J(w, b, x) = \frac{1}{2} \sum_{ (i,j): r(i, j)==1} (w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i, j)})^2$
""")

st.success("This is the special point of collaborative filtering: unlike traditional machine learning algorithms where features must be predefined, collaborative filtering can automatically learn latent features of users and products while learning parameters to predict ratings.")


with st.expander("Explanation"):
    st.info("Reminder: The system will have infinite solutions if the number of hidden variables > number of equations.")
        
    st.write("<h1 style='color:#FF6347'>For Collaborative filtering</h1>", unsafe_allow_html=True)
    st.write("The input to the model is a matrix representing the consumers' ratings of products. This means that each sample will generate more than one equation to link the variables together: ")
    st.markdown("### Case 1: Small data → system has infinite solutions")

    temp1 = {
        "Product": ["A", "B", "C"],
        "User 1": [5, 1, 2],
        "User 2": [0, 3, 4],
        "Feature": ["xA", "xB", "xC"]
    }
    df1 = pd.DataFrame(temp1)
    _, c2, _ = st.columns([1, 1, 1])
    with c2:
        st.table(df1)

    st.write("- For each available rating (there are 6 ratings in total), we have 1 equation in the form: ")
    st.latex(r"w^{(j)} \cdot x^{(i)} = y^{(i, j)}")
    st.write("For User 1, we have weights `w1`. The 3 equations for User 1 are:")

    st.latex(r"w_1 \cdot x_A = 5")
    st.latex(r"w_1 \cdot x_B = 1")
    st.latex(r"w_1 \cdot x_C = 2")
    st.write("- Total number of equations: 6")
    st.write("- Number of hidden variables: ")
    st.markdown("""
    - 3 products → 3 feature variables: `x^(A)`, `x^(B)`, `x^(C)`
    - 2 users → 2 parameter variables: `w^(1)`, `w^(2)`, `b^(1)`, `b^(2)`
    - **Total: 7 hidden variables**
    """)
    st.warning("Number of hidden variables (7) > number of equations (6) → system has infinite solutions")

    st.markdown("---")

    st.markdown("### Case 2: More complete data → system can be determined")

    temp2 = {
        "Product": ["A", "B", "C", "D", "E"],
        "User 1": [5, 4, 2, 3, 0],
        "User 2": [2, 5, 1, 4, 1],
        "Feature": ["xA", "xB", "xC", "xD", "xE"]
    }
    df2 = pd.DataFrame(temp2)
    _, c22, _ = st.columns([1, 1, 1])
    with c22:
        st.table(df2)

    st.write("- Number of equations: 5 products × 2 users = 10 ratings ~ 10 equations")
    st.write("- Number of hidden variables: ")
    st.markdown("""
    - 5 products → 5 feature variables `x^(A)` to `x^(E)`
    - 2 users → 2 variables `w^(1)`, `w^(2)`, `b^(1)`, `b^(2)`
    - **Total: 7 hidden variables**
    """)
    st.success("Number of hidden variables (7) < number of equations (10) → system can be determined (if linearly independent)")

    st.markdown("<h1 style='color:#FF6347'>Traditional Machine Learning Models</h1>", unsafe_allow_html=True)    
    sl_data = pd.DataFrame({
        "x": ["x1", "x2", "x3", "x4", "x5"],
        "y": [3, 5, 7, 9, 11]
    })

    _, c23, _ = st.columns([1, 1, 1])
    with c23:
        st.table(sl_data)

    st.markdown("- Hidden variables: 2 (`w`, `b`) and 5 (`x1` to `x5`)")
    st.markdown("- Equations: 5 (1 for each sample)")
    st.warning("Number of hidden variables (7) > number of equations (5) ⇒ infinite solutions")
    st.write("Even when adding 1 sample, it only adds 1 equation and 1 solution → no matter how much data there is, the number of hidden variables > number of equations")

    st.info("This is because in Collaborative filtering, every time we add a sample (which could be adding a product), \
        the number of equations increases by the number of consumers (users). \
        While the number of hidden variables only increases by the dimension of the feature vector (in this example, it’s 1). \
        So if the data is large enough, the number of equations > number of hidden variables, and it can be solved. \
            However, for supervised learning, it increases linearly, so it always has infinite solutions.")
    
st.markdown('<h2 style="color:#9370DB;">For binary label</h2>', unsafe_allow_html=True)
st.markdown("In practice, we often only care about whether a consumer likes a product or not. That is, we only need to know 1 - Yes and 0 - No, \
    instead of the rating level (from 0-5). Thus, the value of y<sup>(i, j)</sup> for each sample is the probability of Yes, calculated by the formula:", unsafe_allow_html=True)
col1, col2 = st.columns([1.5, 3])
with col2:
    st.markdown(r"""
    $y^{(i,j)} = \sigma(w^{(j)} \cdot x^{(i)} + b^{(j)})$, where $\sigma$ is the function:  $\sigma(z) = \frac{1}{1 + e^{-z}}$
    """)

st.write("Thus, the formula to calculate the probability of whether the consumer likes the product or not is: ")
col1, col2 = st.columns([1.5, 3])
with col2:
    st.markdown(r"""
    $f_{(w, b,, x)}(x) = \sigma(w^{(j)} \cdot x^{(i)} + b^{(j)})$
    """)

st.write("We have the loss function for a sample written as binary cross entropy: ")
col1, col2 = st.columns([1, 3])
with col2:
    st.markdown(r"""
    $L(f_{(w, b,, x)}(x), y^{(i, j)}) = -y^{(i, j)} \cdot log(f_{(w, b,, x)}(x)) - (1-y^{(i, j)}) \cdot log(1-f_{(w, b,, x)}(x)) $
    """)
st.markdown(r"Where the hidden variables to learn are still w, b, x, and y^{(i,j)} is the label taken from the dataset.")
    
st.write("Thus, the total loss for the samples is: ")
col1, col2 = st.columns([1.5, 3])
with col2:
    st.markdown(r"""
    $ J(w, b, x)  = \sum_{ (i,j): r(i, j)==1} L(f_{(w, b,, x)}(x), y^{(i, j)})$
    """)

st.markdown('<h2 style="color:#9370DB;">Applications</h2>', unsafe_allow_html=True)
st.write("User-based: Consumer A likes a few products like Consumer B -> recommend to A what B likes")
st.write("Item-based: Consumer A likes product X, and many people who like X also like Y -> recommend Y to A")
