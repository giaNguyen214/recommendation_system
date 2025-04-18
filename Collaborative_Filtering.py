import streamlit as st 
st.set_page_config(layout="wide", page_title="Collaborative filtering")

import pandas as pd 


st.markdown("""
    <style>
    p {
        font-size: 22px !important;
    }
    li {
        font-size: 22px !important;
    }
    </style>
    
""", unsafe_allow_html=True)


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
data1 = data.drop(columns= ["Price", "Material"])
data2 = data[["Product", "Price", "Material"]]


st.markdown('<h2 style="color:#9370DB;">Tổng quan về giải thuật</h2>', unsafe_allow_html=True)
st.write("Giả sử chúng ta có bảng đánh giá của người tiêu dùng đối với các sản phẩm như thế này (đối với những sản phẩm mà người tiêu dùng chưa đánh giá sẽ kí hiệu là -1)")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.table(data1)
st.write("Một số ký hiệu")
st.markdown( """
    - n<sub>u</sub>: số lượng người tiêu dùng (number of users)
    - n<sub>p</sub>: số lượng sản phẩm (number of products)
    - r(i, j) = 1 nếu người dùng j đã đánh giá sản phẩm i
    - y<sup>(i, j)</sup>: mức đánh giá của người dùng nếu r(i, j) == 1. Nhãn của quá trình huấn luyện
    - Để nhất quán, khi dùng chỉ số i là dành cho sản phẩm, còn j là dành cho người tiêu dùng
""", unsafe_allow_html=True)

st.write("Giả sử đối với mỗi sản phẩm, chúng ta có các đặc trưng x của nó. Ví dụ như giá có tốt hay không và mức độ của nó, 1 là tốt nhất, còn 0 là tệ nhất. Nguyên vật liệu chất lượng hay tệ, màu sắc ưa nhìn không,...")
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.table(data2)
    
st.markdown("Từ đó, ta có định nghĩa feature vector x<sup>(i)</sup> cho mỗi sản phẩm, ví dụ:", unsafe_allow_html=True)
st.markdown("""
            - AS - NÓN LƯỠI TRAI SỐNG ĐẦY x<sup>(1)</sup> = [0.9, 0]
            - Kính lão nam nữ không viền: x<sup>(2)</sup> = [1, 0.01]
            """, unsafe_allow_html=True)

st.markdown("Chúng ta cần xây dựng một model với các tham số w<sup>(j)</sup> và b<sup>(j)</sup> để dự đoán rating của người tiêu dùng đổi với sản phẩm dựa trên feature vector của sản phẩm đó: rating ≈ w<sup>(j)</sup>.x<sup>(i)</sup> + b<sup>(j)</sup>. Và xác định loss function cho bài toán như sau", unsafe_allow_html=True)

st.write("Để dự đoán rating của người tiêu dùng j cho tất cả sản phẩm:")
col1, col2 = st.columns([1.5, 3])
with col2:
    st.markdown(r"""
$J(w^{(j)}, b^{(j)}) = \frac{1}{2} \sum_{i: r(i, j)==1} (w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i, j)})^2$
""")
st.markdown("Trong đó, y<sup>(i, j)</sup> là giá trị rating từ dữ liệu chúng ta đã có được, dùng nó làm nhãn để huấn luyện và đánh giá độ chính xác mô hình. \
    Còn w<sup>(j)</sup>.x<sup>(i)</sup> + b<sup>(j)</sup> là kết quả dự đoán của mô hình. \
        Và điều kiện i: r(i, j) == 1 là để lọc ra, chỉ xét những sản phẩm i mà người tiêu dùng j đã đánh giá \
        (vì chỉ những sản phẩm đó mới có y<sup>(i, j)</sup> để huấn luyện mô hình)", unsafe_allow_html=True)
st.write("Tương tự, để tìm tham số cho tất cả các người dùng khác thì chúng ta tính loss function cho toàn bộ người tiêu dùng: ")
col1, col2 = st.columns([1.5, 3])
with col2:
    st.markdown(r"""
    $J(w^{(1)}...w^{(n_{u})}, b^{(1)}...b^{(n_{u})}) = \sum_{j=1}^{n_{u}} J(w^{(j)}, b^{(j)})$
    """)
st.write("Việc còn lại là minimize loss function, từ đó tìm ra các tham số tối ưu cho mô hình")
st.info("Đến hiện tại thì bài toán vẫn tương tự như supervised learning")

st.markdown('<h2 style="color:#9370DB;">Vấn đề nếu không có feature vector của sản phẩm</h2>', unsafe_allow_html=True)
temp = data.copy()
temp["Price"] = [-1]*5
temp["Material"] = [-1]*5
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.table(temp)

st.write("Tương tự ở trên khi mà chúng ta giả sử có feature vector x và đi tìm w, b. Bây giờ chúng ta giả sử rằng biết trước các giá trị của w và b và đi tìm x.")
col1, col2 = st.columns([1.5, 3])
with col2:
    st.markdown(r"""
$J(x^{(i)}) = \frac{1}{2} \sum_{j: r(i, j)==1} (w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i, j)})^2$
""")
st.write("Tương tự, tìm đặc trưng cho tất cả sản phẩm khác: ")
col1, col2 = st.columns([1.5, 3])
with col2:
    st.markdown(r"""
    $J(x^{(1)}...x^{(n_{p})}) = \sum_{i=1}^{n_{p}} J(x^{(i)})$
    """)

st.markdown('<h2 style="color:#9370DB;">Tổng hợp</h2>', unsafe_allow_html=True)
st.markdown(r"Từ đó, chúng ta gom 2 công thức lại, loss function trở thành hàm của các biến $w^{(1)}...w^{(n_{u})}$, $b^{(1)}...b^{(n_{u})}$ và $x^{(1)}...x^{(n_{p})}$") # chú ý chỗ này x là feature VECTOR chứ không phải chỉ 1 giá trị nha
col1, col2 = st.columns([1.5, 3])
with col2:
    st.markdown(r"""
    $J(w, b, x) = \frac{1}{2} \sum_{ (i,j): r(i, j)==1} (w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i, j)})^2$
    """)



st.success("Đây là điểm đặc biệt của collaborative filtering: khác với các thuật toán machine learning truyền thống, khi mà các đặc trưng (features) \
    phải được xác định và cung cấp trước. \
    Collaborative filtering có thể tự động học được các đặc trưng tiềm ẩn của người dùng và sản phẩm, đồng thời học tham số để dự đoán mức độ \
    đánh giá (rating).")

with st.expander("Giải thích"):
    st.info("Nhắc lại: hệ sẽ vô số nghiệm nếu số ẩn > số phương trình")
        
    st.write("<h1 style='color:#FF6347'>Đối với Collaborative filtering</h1>", unsafe_allow_html=True)
    st.write("Đầu vào của mô hình là một ma trận biểu diễn đánh giá của người tiêu dùng đối với sản phẩm. Nghĩa là mỗi sample sẽ sinh ra nhiều hơn 1 phương trình để liên kết các biến với nhau: ")
    st.markdown("### Trường hợp 1: Dữ liệu nhỏ → hệ vô số nghiệm")

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

    st.write("- Với mỗi rating có sẵn (có tổng cộng 6 rating), ta có 1 phương trình dạng:")
    st.latex(r"w^{(j)} \cdot x^{(i)} = y^{(i, j)}")
    st.write("- Tổng số phương trình: 6")
    st.write("- Số ẩn:")
    st.markdown("""
    - 3 sản phẩm → 3 biến đặc trưng: `x^(A)`, `x^(B)`, `x^(C)`
    - 2 người dùng → 2 biến tham số: `w^(1)`, `w^(2)`, `b^(1)`, `b^(2)`
    - **Tổng: 7 ẩn**
    """)
    st.warning("Số ẩn (7) > số phương trình (6) → hệ vô số nghiệm")

    st.markdown("---")

    st.markdown("### Trường hợp 2: Dữ liệu đầy đủ hơn → hệ có thể xác định")

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

    st.write("- Số phương trình: 5 sản phẩm × 2 người dùng = 10 ratings ~ 10 phương trình")
    st.write("- Số ẩn:")
    st.markdown("""
    - 5 sản phẩm → 5 biến đặc trưng `x^(A)` đến `x^(E)`
    - 2 người dùng → 2 biến `w^(1)`, `w^(2)`, `b^(1)`, `b^(2)`
    - **Tổng: 7 ẩn**
    """)
    st.success("Số ẩn (7) < số phương trình (10) → hệ có thể xác định (nếu độc lập tuyến tính)")

    
    
    st.markdown("<h1 style='color:#FF6347'>Các mô hình Machine learning truyền thống</h1>", unsafe_allow_html=True)    
    sl_data = pd.DataFrame({
        "x": ["x1", "x2", "x3", "x4", "x5"],
        "y": [3, 5, 7, 9, 11]
    })

    _, c23, _ = st.columns([1, 1, 1])
    with c23:
        st.table(sl_data)

    st.markdown("- Ẩn số: 2 (`w`, `b`) và 5 (`x1` đến `x5`)")
    st.markdown("- Phương trình: 5 (1 cho mỗi sample)")
    st.warning("Số ẩn (7) > số phương trình (5) ⇒ vô số nghiệm")
    st.write("Môi khi thêm 1 sample vào thì chỉ thêm 1 phương trình và 1 nghiệm -> dù dữ liệu có nhiều đến đâu thì số ẩn > số phương trình")


    st.info("Điều này là vì Collaborative filtering, mỗi khi thêm 1 sample vào (có thể là thêm một product) \
        thì số phương trình sẽ tăng thêm bằng với số lượng người tiêu dùng (user). \
        Trong khi số ẩn chỉ tăng lên bằng với số chiều của feature vector (trong ví dụ trên là 1). \
        Cho nên nếu dữ liệu đủ nhiều thì số phương trình > số ẩn, có thể giải được. \
            Còn đối với supervised thông thường thì nó tăng tuyến tính nên luôn vô số nghiệm.")
    
st.markdown('<h2 style="color:#9370DB;">Đối với binary label</h2>', unsafe_allow_html=True)
st.markdown("Trong thực tế, chúng ta thường chỉ quan tâm đến người tiêu dùng có thích sản phẩm đó hay không? Nghĩa là chỉ cần biết 1 - Yes và 0 - No\
    chứ không cần mức độ đánh giá (rating) từ 0-5. Như vậy, giá trị y<sup>(i, j)</sup> của mỗi sample là xác suất Yes, được tính bằng công thức:", unsafe_allow_html=True)
col1, col2 = st.columns([1.5, 3])
with col2:
    st.markdown(r"""
    $y^{(i,j)} = \sigma(w^{(j)} \cdot x^{(i)} + b^{(j)})$, với $\sigma$ là hàm:  $\sigma(z) = \frac{1}{1 + e^{-z}}$
    """)

st.write("Như vậy, công thức để tính xác suất người tiêu dùng có thích sản phẩm hay không là:")
col1, col2 = st.columns([1.5, 3])
with col2:
    st.markdown(r"""
    $f_{(w, b,, x)}(x) = \sigma(w^{(j)} \cdot x^{(i)} + b^{(j)})$
    """)

st.write("Ta có loss function cho một sample được viết theo binary cross entropy:")
col1, col2 = st.columns([1, 3])
with col2:
    st.markdown(r"""
    $L(f_{(w, b,, x)}(x), y^{(i, j)}) = -y^{(i, j)} \cdot log(f_{(w, b,, x)}(x)) - (1-y^{(i, j)}) \cdot log(1-f_{(w, b,, x)}(x)) $
    """)
st.markdown(r"Với các ẩn cần học vẫn là w, b, x còn y^{(i,j)} là nhãn được lấy từ dataset")
    
st.write("Vậy ta có tổng loss cho các samples là:")
col1, col2 = st.columns([1.5, 3])
with col2:
    st.markdown(r"""
    $ J(w, b, x)  = \sum_{ (i,j): r(i, j)==1} L(f_{(w, b,, x)}(x), y^{(i, j)})$
    """)

st.markdown('<h2 style="color:#9370DB;">Ứng dụng</h2>', unsafe_allow_html=True)
st.write("User-based: người tiêu dùng A thích một vài sản phẩm giống người tiêu dùng B -> gợi ý cho A những gì B thích")
st.write("Item-based: người tiêu dùng A thích sản phẩm X và nhiều người thích X cũng thích Y -> gợi ý Y cho A")