import streamlit as st 
st.set_page_config(layout="wide", page_title="Content-based filtering")

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

# st.markdown('<h2 style="color:#A3E4D7;">Overview of the Algorithm</h2>', unsafe_allow_html=True)
st.markdown('<h2 style="color:#2ECC71;">Overview of the Algorithm</h2>', unsafe_allow_html=True)
st.markdown("""
    - Collaborative filtering: Recommends products based on the ratings of other consumers with similar ratings to you.
    - Content-based filtering: Recommends products based on the characteristics of the consumer and the product.
""")
col1, _, col2 = st.columns(3)
with col1:
    st.markdown(r"User features: $x_{u}^{(j)}$ for consumer j")
    st.markdown("""
    - Age
    - Gender
    - Products purchased
    - Average rating for each category (e.g., average rating for books is 4.5, but clothing is rated 2.3)
    - ...
    """)
with col2:
    st.markdown(r"Product features: $x_{p}^{(i)}$ for product i")
    st.markdown("""
    - Year of production
    - Type? (Book, Clothing, Household goods, ...)
    - Consumer reviews
    - Average rating received
    - ...
    """)
col1, col2 = st.columns([1, 2])
with col2:
    st.markdown(r"To predict the rating of consumer j for product i:  $v_{u}^{(j)} \cdot v_{p}^{(i)}$")
col1, col2 = st.columns([0.9, 2])
with col2:
    st.markdown(r"Where $v_{u}^{(j)}$ is calculated from $x_{u}^{(j)}$, $v_{p}^{(i)}$ is calculated from $x_{p}^{(i)}$, and $\cdot$ is the dot product operation")
st.markdown(r"""
Intuition: 
- $v_{u}^{(j)} = [4.9, 0.1, 3]$: consumer j prefers products with:
    - Good price: 4.9
    - Quality materials: 0.1 (not very important)
    - Trendy color: 3

- $v_{p}^{(i)} = [2, 3, 1.4]$: product i has:
    - Price: 2
    - Material: 3
    - Color: 1.4
""")
st.markdown('<h2 style="color:#2ECC71;">Predicting Rating</h2>', unsafe_allow_html=True)
st.markdown(r"From the initial features $x_{u}^{(j)}$ and $x_{p}^{(i)}$, we can use a simple neural network to learn the feature vectors $v_{u}^{(j)}$ and $v_{p}^{(i)}$ respectively:")
col1, col2 = st.columns([1, 10])
with col2:
    st.image("images/neural_net.jpg")
st.info(r"$x_{u}^{(j)}$ and $x_{p}^{(i)}$ may have different sizes. However, after passing through the neural network, they need to have the same size (here 32) to perform the dot product.")
st.warning("Therefore, the two branches can differ in the number of layers and neurons in each layer, but the number of neurons in the final layer must be the same.")

st.markdown(r"If you want to predict the rating of consumer j for product i, calculate $v_{u}^{(j)} \cdot v_{p}^{(i)}$. For binary labels, apply the sigmoid function: $\sigma (v_{u}^{(j)} \cdot v_{p}^{(i)})$. Then calculate the loss and optimize to find the appropriate parameters.")

st.markdown('<h2 style="color:#2ECC71;">Retrieval and Ranking</h2>', unsafe_allow_html=True)
st.write("If the dataset is too large, a few steps need to be taken to speed up the processing.")
st.write("Retrieval: ")
st.markdown("""
    - Pre-create a list of candidate products for recommendation, for example:
        - Among the last 10 products the user rated, search for similar products.
        - Among the 3 product categories the user just viewed, find the top 10 best products.
        - Find the top 20 best-selling products.
    - Combine all the products found, remove duplicates and those already rated by the consumer.
""")

st.write("Ranking: ")
st.markdown("""
    - When a consumer rates new products, more data becomes available, and the model needs to be retrained.
    - However, instead of retraining on the entire dataset (which is time-consuming and costly), only the data retrieved in the previous step is used for re-ranking.
""")
