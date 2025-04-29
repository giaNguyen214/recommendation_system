
import torch.nn as nn
import numpy as np
import torch
import joblib
import pickle
import pandas as pd
import streamlit as st

st.set_page_config(layout='wide')

# Reload the CSV
rating = pd.read_csv("models/rating.csv")

# Reload encoders
customer_encoder = joblib.load("models/customer_encoder.pkl")
product_encoder = joblib.load("models/product_encoder.pkl")

# Reload the product mean with pickle
with open("models/product_mean.pkl", "rb") as f:
    product_mean = pickle.load(f)

n_customers, n_products, n_categories = 142829, 5809, 3
    

class LinearUserItemModel(nn.Module):
    def __init__(self, n_users, n_items, n_categories, feat_dim=64, cat_dim=16):
        super().__init__()
        self.user_w     = nn.Embedding(n_users, feat_dim)
        self.user_b     = nn.Embedding(n_users, 1)
        self.item_x     = nn.Embedding(n_items, feat_dim)
        self.category_x = nn.Embedding(n_categories, cat_dim)

        # nếu cat_dim != feat_dim, cần map cat_dim -> feat_dim trước khi cộng
        if cat_dim != feat_dim:
            self.cat2item = nn.Linear(cat_dim, feat_dim, bias=False)
        else:
            self.cat2item = None

    def forward(self, user_idx, item_idx, category_idx):
        w_u = self.user_w(user_idx)             # (batch, feat_dim)
        b_u = self.user_b(user_idx).squeeze()   # (batch,)
        x_i = self.item_x(item_idx)             # (batch, feat_dim)
        c_i = self.category_x(category_idx)     # (batch, cat_dim)

        # nếu cần, chiếu c_i lên không gian feat_dim
        if self.cat2item is not None:
            c_i = self.cat2item(c_i)

        combined = x_i + c_i                    # (batch, feat_dim)
        pred = (w_u * combined).sum(dim=1) + b_u # (batch,)
        return pred
    
  


model = LinearUserItemModel(
    n_customers, n_products, n_categories,
    feat_dim=64, cat_dim=16
)
model.load_state_dict(torch.load("models/recommendation_system_model.pt", weights_only=True))
model.eval()

# 1. Chuẩn bị các mapping cần thiết (chạy sau khi load xong rating và encoder):
n_products = product_encoder.classes_.shape[0]

# Danh sách raw product_id theo thứ tự index của product_encoder
product_idx_to_raw = {
    idx: raw_id
    for idx, raw_id in enumerate(product_encoder.inverse_transform(np.arange(n_products)))
}

# Map từ productID (encoded) -> categoryID (encoded)
product_to_cat = (
    rating
    .drop_duplicates(subset=['productID'])
    .set_index('productID')['categoryID']
    .to_dict()
)

# Tạo tensor item_means theo thứ tự productIndex
item_means_tensor = torch.tensor(
    [ product_mean[ product_idx_to_raw[i] ] for i in range(n_products) ],
    dtype=torch.float32
)

# 2. Định nghĩa hàm recommend
def get_top_k_recommendations(
    model,
    raw_user_id: str,
    k: int = 10,
    customer_encoder=customer_encoder,
    product_encoder=product_encoder,
    product_idx_to_raw=product_idx_to_raw,
    product_to_cat=product_to_cat,
    item_means_tensor=item_means_tensor,
    device: str = 'cpu'
):
    """
    Trả về list [(raw_product_id, predicted_rating), ...] top k cho user raw_user_id.
    """

    # 2.1. Kiểm tra user
    if raw_user_id not in customer_encoder.classes_:
        raise ValueError(f"Unknown user_id: {raw_user_id}")

    # 2.2. Encode user
    user_idx = int(customer_encoder.transform([raw_user_id])[0])

    # 2.3. Tạo tensor users/items/categories
    all_item_idxs = torch.arange(n_products, device=device)
    all_user_idxs = torch.full(
        (n_products,), user_idx, dtype=torch.long, device=device
    )
    all_cat_idxs = torch.tensor(
        [ product_to_cat[i.item()] for i in all_item_idxs ],
        dtype=torch.long, device=device
    )

    # 2.4. Dự đoán centered rating
    model.eval()
    with torch.no_grad():
        pred_centered = model(
            all_user_idxs,
            all_item_idxs,
            all_cat_idxs
        ).cpu()

    # 2.5. Cộng lại mean rating để được rating cuối
    pred_full = pred_centered + item_means_tensor

    # 2.6. Lấy top k
    topk_vals, topk_idxs = torch.topk(pred_full, k)

    # 2.7. Build kết quả raw
    results = []
    for score, idx in zip(topk_vals.tolist(), topk_idxs.tolist()):
        raw_pid = product_idx_to_raw[idx]
        results.append((raw_pid, score))

    return results

def recommend_for_user(
    raw_user_id: str,
    user_ratings: dict,      # { raw_product_id: raw_rating, ... }
    model: nn.Module,
    product_encoder,
    product_to_cat: dict,
    product_idx_to_raw: dict,
    item_means_tensor: torch.Tensor,
    product_mean: dict,
    k: int = 10,
    device: str = 'cpu'
):
    """
    Trả về list [(raw_pid, score), ...] top-k
    - user_ratings: nếu user mới, có thể {} hoặc có vài entry
    - nếu raw_user_id in encoder => gọi collaborative filtering cũ
    """
    
    # 2. Nếu user mới mà không có rating -> khỏi chạy vì nó chỉ trả về product mean thôi
    if not user_ratings:
        # Top-k product_mean
        popular = sorted(product_mean.items(), key=lambda x: x[1], reverse=True)
        return popular[:k]

    # 3. User mới với một vài rating => build pseudo-user embedding
    model.to(device)
    model.eval()

    # Lấy idx và centered rating của các item đã đánh giá
    rated_raw = list(user_ratings.keys())
    rated_idxs = [ int(product_encoder.transform([pid])[0]) for pid in rated_raw ]
    centered = [ user_ratings[pid] - product_mean[int(pid)] for pid in rated_raw ]  # rating - mean

    # Tạo tensor
    idxs_t = torch.tensor(rated_idxs, device=device)
    center_t = torch.tensor(centered, dtype=torch.float32, device=device).unsqueeze(1)

    # Embedding item và category
    x_i = model.item_x(idxs_t)                  # (n_rated, feat_dim)
    c_i = model.category_x(torch.tensor([product_to_cat[i] for i in rated_idxs], device=device))
    if model.cat2item:
        c_i = model.cat2item(c_i)               # (n_rated, feat_dim)
    combined_rated = x_i + c_i                  # (n_rated, feat_dim)

    # Tính vector user: weighted average by centered rating
    user_vec = (combined_rated * center_t).mean(dim=0)  # (feat_dim,)

    # Score tất cả items
    all_idxs = torch.arange(item_means_tensor.shape[0], device=device)
    all_x = model.item_x(all_idxs)
    all_c = model.category_x(torch.tensor([product_to_cat[i.item()] for i in all_idxs], device=device))
    if model.cat2item:
        all_c = model.cat2item(all_c)
    combined_all = all_x + all_c   # (n_products, feat_dim)

    # Dot product
    scores = combined_all @ user_vec  # (n_products,)
    scores = scores.cpu() + item_means_tensor  # cộng mean

    # Loại bỏ các item đã đánh giá
    scores[rated_idxs] = -1e9

    # Lấy top-k
    topk_vals, topk_idxs = torch.topk(scores, k)
    results = [(product_idx_to_raw[int(idx)], float(score)) for score, idx in zip(topk_vals, topk_idxs)]
    return results


# user_ratings = {
#     "276250848": 4.0,
#     "277776275": 5.0,
# }
# top_n = 50
# recs = recommend_for_user(
#     raw_user_id="NEW_USER_1",
#     user_ratings=user_ratings,        
#     model=model,
#     product_encoder=product_encoder,
#     product_to_cat=product_to_cat,
#     product_idx_to_raw=product_idx_to_raw,
#     item_means_tensor=item_means_tensor,
#     product_mean=product_mean,
#     k=top_n
# )



















# Khởi tạo feedbacks trong session_state nếu chưa có
if 'feedbacks' not in st.session_state:
    st.session_state['feedbacks'] = {}


df = pd.read_csv("models/Product.csv")

st.title("Product Information and Rating")

st.write("Assume you are a new customer. You purchase a few items and give feedback (ratings: 1-5 stars). Then click 'Submit Feedbacks' and 'Show recommended product' — based on the feedback you just provided, the system will suggest some products that match your preferences. Alternatively, if you don't rate any products, the system will recommend the top-rated products to you.")

# Phân trang
PAGE_SIZE = 20
page_number = st.number_input("Page", min_value=1, max_value=(len(df) - 1)//PAGE_SIZE + 1, value=1)
start_idx = (page_number - 1) * PAGE_SIZE
end_idx = start_idx + PAGE_SIZE
page_data = df.iloc[start_idx:end_idx]

# Header bảng
st.subheader(f"Products (Page {page_number})")
cols = st.columns([1, 3, 2, 2, 2, 2, 2, 2])
headers = ['ID', 'Name', 'Brand', 'Price', 'Discount', 'Rating Avg', 'Review Count', 'Your Feedback']
for col, header in zip(cols, headers):
    col.markdown(f"**{header}**")


for idx, row in page_data.iterrows():
    cols = st.columns([1, 3, 2, 2, 2, 2, 2, 2])

    cols[0].write(row['id'])
    cols[1].write(row['name'])
    cols[2].write(row['brand_name'])
    cols[3].write(f"{row['price']} VND")
    cols[4].write(f"{row['discount_rate']}%")
    cols[5].write(f"{row['rating_average']}")
    cols[6].write(row['review_count'])

    # Lấy feedback cũ (nếu có)
    default = st.session_state['feedbacks'].get(row['id'], None)

    # Thêm feedback component, key phải unique toàn app
    fb = cols[7].feedback(
        "stars",
        key=f"feedback_{row['id']}",
    )
    # Lưu vào session_state luôn
    st.session_state['feedbacks'][row['id']] = fb + 1 if fb is not None else fb

# Khi submit, show bảng feedbacks
if st.button("Submit Feedbacks"):
    feedback_df = pd.DataFrame([
        {"Product ID": pid,
         "Product Name": df.loc[df['id']==pid, 'name'].iloc[0],
         "Feedback": fb}
        for pid, fb in st.session_state['feedbacks'].items()
        if fb is not None
    ])
    st.subheader("Your Feedbacks (All Pages)")
    st.dataframe(feedback_df)
    
if st.button('Show recommended product:'):
    # Lấy các đánh giá đã có
    user_ratings = {
        key: val
        for key, val in st.session_state['feedbacks'].items()
        if val is not None
    }

    # Lấy danh sách gợi ý
    top_n = 50
    recs = recommend_for_user(
        raw_user_id="NEW_USER_1",
        user_ratings=user_ratings,
        model=model,
        product_encoder=product_encoder,
        product_to_cat=product_to_cat,
        product_idx_to_raw=product_idx_to_raw,
        item_means_tensor=item_means_tensor,
        product_mean=product_mean,
        k=top_n
    )

    # Hiển thị sản phẩm đã đánh giá
    st.title("Rated Products")
    rated_df = pd.DataFrame(user_ratings.items(), columns=['Product ID', 'Rating'])
    rated_df['Product Name'] = rated_df['Product ID'].apply(lambda x: df.loc[df['id'] == int(x), 'name'].values[0])
    rated_df = rated_df[['Product Name', 'Rating']]
    st.dataframe(rated_df)

    # Hiển thị gợi ý
    st.title(f"Top {top_n} Recommended Products")
    recs_df = pd.DataFrame(recs, columns=['Product ID', 'Predicted Rating'])
    recs_df['Product Name'] = recs_df['Product ID'].apply(lambda x: df.loc[df['id'] == x, 'name'].values[0])
    recs_df = recs_df[['Product Name', 'Predicted Rating']]
    st.dataframe(recs_df)


st.subheader("🔎 Search Product by ID")

search_product_id = st.text_input("Enter Product ID to Search")

if search_product_id:
    # Tìm sản phẩm theo id
    search_result = df[ df['id'] == int(search_product_id)]
    # print("search_product_id: ", search_product_id, "type search_product_id: ", type(search_product_id))
    # print(search_result)
    # print(df.shape)
    # print(int(search_product_id) == 192135155)
    # print(type(df['id']))
    if not search_result.empty:
        # Nếu tìm thấy, hiện ra bảng format giống trang list
        st.write("### Search Result:")

        for idx, row in search_result.iterrows():
            cols = st.columns([1, 3, 2, 2, 2, 2, 2, 2])

            cols[0].write(row['id'])
            cols[1].write(row['name'])
            cols[2].write(row['brand_name'])
            cols[3].write(f"${row['price']}")
            cols[4].write(f"{row['discount_rate'] * 100:.1f}%")
            cols[5].write(f"{row['rating_average']}")
            cols[6].write(row['review_count'])

            # Lấy feedback cũ nếu có
            default = st.session_state['feedbacks'].get(row['id'], None)

            # Feedback component (key phải unique)
            fb = cols[7].feedback(
                "stars",
                key=f"search_feedback_{row['id']}"
            )

            # Lưu feedback vào session state
            st.session_state['feedbacks'][row['id']] = fb + 1 if fb is not None else fb
    else:
        st.warning("⚠️ Product not found.")

    

