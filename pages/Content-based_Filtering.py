import streamlit as st 
st.set_page_config(layout="wide", page_title="Collaborative filtering")


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

# st.markdown('<h2 style="color:#A3E4D7;">Tổng quan về giải thuật</h2>', unsafe_allow_html=True)
st.markdown('<h2 style="color:#2ECC71;">Tổng quan về giải thuật</h2>', unsafe_allow_html=True)
st.markdown("""
    - Collaborative filtering: đề xuất sản phẩm dựa trên rating của những người tiêu dùng khác có rating giống mình
    - Content-based filtering: đề xuất sản phẩm dựa trên đặc trưng của người tiêu dùng và đặc trưng sản phẩm
""")
col1, _,  col2 = st.columns(3)
with col1:
    st.markdown(r"User features: $x_{u}^{(j)}$ cho người tiêu dùng j")
    st.markdown("""
    - Tuổi
    - Giới tính
    - Những sản phẩm đã mua
    - Trung bình đánh giá mỗi thể loại (vd: trung bình đánh giá sách 4.5 nhưng chỉ đánh giá mặc hàng quần áo 2.3)
    - ...
    """)
with col2:
    st.markdown(r"Product features: $x_{p}^{(i)}$ cho sản phẩm i")
    st.markdown("""
    - Năm sản xuất
    - Loại? (Sách, Quần áo, Đồ gia dụng, ...)
    - Reviews của người tiêu dùng
    - Trung bình đánh giá nhận được
    - ...
    """)
col1, col2 = st.columns([1, 2])
with col2:
    st.markdown(r"Để dự đoán rating của người dùng j cho sản phẩm i:  $v_{u}^{(j)} \cdot v_{p}^{(i)}$")
col1, col2 = st.columns([0.9, 2])
with col2:
    st.markdown(r"Với $v_{u}^{(j)}$ được tính từ $x_{u}^{(j)}$, $v_{p}^{(i)}$ được tính từ $x_{p}^{(i)}$ và $\cdot$ là phép dot product")
st.markdown(r"""
Trực giác: 
- $v_{u}^{(j)} = [4.9, 0.1, 3]$: người dùng j thích sản phẩm có:
    - Giá cả tốt: 4.9
    - Nguyên vật liệu chất lượng: 0.1 (ít quan tâm)
    - Màu sắc thịnh hành: 3

- $v_{p}^{(i)} = [2, 3, 1.4]$: sản phẩm i có:
    - Giá cả: 2
    - Nguyên vật liệu: 3
    - Màu sắc: 1.4
""")
st.markdown('<h2 style="color:#2ECC71;">Dự đoán rating</h2>', unsafe_allow_html=True)
st.markdown(r"Từ đặc trưng ban đầu $x_{u}^{(j)}$ và $x_{p}^{(i)}$, chúng ta có thể dùng mạng neural đơn giản để học ra vector đặc trưng $v_{u}^{(j)}$ và $v_{p}^{(i)}$ tương ứng:")
col1, col2 = st.columns([1, 10])
with col2:
    st.image("images/neural_net.jpg")
st.info(r"$x_{u}^{(j)}$ và $x_{p}^{(i)}$ có thể có size khác nhau. Nhưng sau khi qua Neural network thì chúng cần phải cùng size (ở đây là 32) để có thể thực hiện dot product")
st.warning("Do đó, 2 nhánh có thể khác nhau về số layers, về số neurons ở các layer, tuy nhiên, số neurons ở layer cuối cùng phải bằng nhau")

st.markdown(r"Nếu muốn dự đoán rating của người tiêu dùng j cho sản phẩm i thì tính $v_{u}^{(j)} \cdot v_{p}^{(i)}$. Trong trường hợp binary label thì đưa kết quả qua hàm sigmoid: $\sigma (v_{u}^{(j)} \cdot v_{p}^{(i)})$. Và thực hiện tính loss và đi tối ưu để tìm tham số phù hợp")

# st.markdown('<h2 style="color:#2ECC71;">Retrieval và ranking</h2>', unsafe_allow_html=True)
# st.write("Nếu dữ liệu quá lớn, cần phải thực hiện trước một vài bước để tăng tốc xử lý")
# st.write("Retrieval: ")
# st.markdown("""
#     - Tạo trước một danh sách các ứng viên để đề xuất, ví dụ:
#         - Trong 10 sản phẩm gần nhất mà người dùng đánh giá, thực hiện tìm kiếm các sản phẩm tương tự
#         - Trong 3 danh mục sản phẩm mà người dùng vừa xem, tìm 10 sản phẩm tốt nhất
#         - Tìm top 20 sản phẩm bán chạy nhất
#     - Kết hợp tất cả sản phẩm vừa tìm được, xóa các duplicates và những sản phẩm mà người tiêu dùng đã đánh giá.
# """)

# st.write("Ranking: ")
# st.markdown("""
#     - Khi người tiêu dùng đánh giá những sản phẩm mới -> có thêm dữ liệu -> cần chạy lại model
#     - Tuy nhiên, không chạy lại trên toàn bộ dataset (tốn thời gian và chi phí) mà chỉ tính trên data vừa retrieved được ở bước trên
# """)