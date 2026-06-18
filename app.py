import streamlit as st
# Import thư viện Streamlit và đặt tên ngắn là "st"
# Streamlit giúp tạo giao diện web bằng Python

st.title("🍱 Food Waste Rescue Radar")
# Hiển thị tiêu đề lớn ở đầu trang web

event_type = st.selectbox(
    "Event Type",
    ["School Lunch", "Community Event"]
)
# Tạo hộp lựa chọn (dropdown)
# Dòng đầu là câu hỏi hiển thị
# Danh sách bên dưới là các lựa chọn
# Giá trị người dùng chọn sẽ được lưu vào biến event_type

location = st.selectbox(
    "Location",
    ["Cafeteria", "Event Venue"]
)
# Tạo dropdown thứ hai
# Cho người dùng chọn địa điểm

expected = st.number_input(
    "Expected Attendance",
    min_value=0
)
# Tạo ô nhập số
# Người dùng nhập số người dự kiến tham gia
# min_value=0 nghĩa là không được nhập số âm
# Giá trị nhập được lưu vào biến expected

prepared = st.number_input(
    "Food Prepared",
    min_value=0
)
# Tạo ô nhập số lượng suất ăn đã chuẩn bị
# Kết quả lưu vào biến prepared

weather = st.selectbox(
    "Weather",
    ["Sunny", "Rainy", "Cloudy"]
)
# Dropdown chọn thời tiết
# Sunny = nắng
# Rainy = mưa
# Cloudy = nhiều mây

st.button("Predict Waste")
# Tạo nút bấm có tên "Predict Waste"
# Hiện tại nút này chưa làm gì cả
# Chỉ mới xuất hiện trên giao diện
