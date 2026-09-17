import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# =========================
# ĐỌC DỮ LIỆU
# =========================
data = pd.read_csv("Linear_gression_KFOLD/data.csv")

print("========== DỮ LIỆU ==========")
print(data)

print("\nKích thước dữ liệu:")
print(data.shape)


X = data[
    [
        "dien_tich",
        "so_phong",
        "khoang_cach"
    ]
]

y = data["gia"]


# =========================
# CHIA TRAIN / TEST
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n========== TRAIN / TEST ==========")
print("Số dữ liệu train:", len(X_train))
print("Số dữ liệu test:", len(X_test))


# =========================
# TẠO MÔ HÌNH OVERFITTING
# =========================

degree_overfit = 8

overfit_model = make_pipeline(
    PolynomialFeatures(degree=degree_overfit),
    LinearRegression()
)

overfit_model.fit(X_train, y_train)


# Dự đoán trên tập Train và Test

y_train_pred = overfit_model.predict(X_train)
y_test_pred = overfit_model.predict(X_test)


# Tính MSE

train_mse_before = mean_squared_error(
    y_train,
    y_train_pred
)

test_mse_before = mean_squared_error(
    y_test,
    y_test_pred
)
# =========================
# K-FOLD CROSS VALIDATION
# =========================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

degrees = range(1, 9)

results = []


print("\n========== K-FOLD CROSS VALIDATION ==========")

for degree in degrees:

    model = make_pipeline(
        PolynomialFeatures(degree=degree),
        LinearRegression()
    )

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=kf,
        scoring="neg_mean_squared_error"
    )

    mean_mse = -scores.mean()

    results.append({
        "Degree": degree,
        "MSE": mean_mse
    })

    print(
        "Degree:",
        degree,
        "-> CV MSE:",
        round(mean_mse, 4)
    )


# =========================
# CHỌN DEGREE TỐT NHẤT
# =========================

results_df = pd.DataFrame(results)

best_degree = results_df.loc[
    results_df["MSE"].idxmin(),
    "Degree"
]

best_cv_mse = results_df["MSE"].min()


print("\n========== KẾT QUẢ K-FOLD ==========")
print("Best degree:", best_degree)
print("Best CV MSE:", round(best_cv_mse, 4))


# =========================
# TẠO MÔ HÌNH TỐT NHẤT
# =========================

fixed_model = make_pipeline(
    PolynomialFeatures(
        degree=int(best_degree)
    ),
    LinearRegression()
)

fixed_model.fit(X_train, y_train)


# =========================
# ĐÁNH GIÁ MÔ HÌNH TỐT NHẤT
# =========================

y_train_pred_fixed = fixed_model.predict(X_train)
y_test_pred_fixed = fixed_model.predict(X_test)


train_mse_after = mean_squared_error(
    y_train,
    y_train_pred_fixed
)

test_mse_after = mean_squared_error(
    y_test,
    y_test_pred_fixed
)


print("\n========== SAU KHI DÙNG K-FOLD ==========")
print("Best degree:", best_degree)
print("Train MSE:", round(train_mse_after, 4))
print("Test MSE :", round(test_mse_after, 4))
# =========================
# TRAIN FINAL MODEL
# =========================

final_model = make_pipeline(
    PolynomialFeatures(
        degree=int(best_degree)
    ),
    LinearRegression()
)

final_model.fit(X, y)


# =========================
# NHẬP THÔNG TIN NHÀ MỚI
# =========================

print("\n========== NHẬP DỮ LIỆU MỚI ==========")

dien_tich = float(
    input("Nhập diện tích: ")
)

so_phong = int(
    input("Nhập số phòng: ")
)

khoang_cach = float(
    input("Nhập khoảng cách: ")
)


new_house = pd.DataFrame({
    "dien_tich": [dien_tich],
    "so_phong": [so_phong],
    "khoang_cach": [khoang_cach]
})


# =========================
# DỰ ĐOÁN GIÁ NHÀ
# =========================

predicted_price = final_model.predict(
    new_house
)


print("\n========== KẾT QUẢ DỰ ĐOÁN ==========")

print(
    "Best degree:",
    best_degree
)

print(
    "Giá nhà dự đoán:",
    round(predicted_price[0], 4),
    "tỷ VNĐ"
)
