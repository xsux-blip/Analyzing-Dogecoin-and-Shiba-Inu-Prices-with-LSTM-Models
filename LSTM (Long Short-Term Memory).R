# Instalasi library yang diperlukan jika belum terpasang
# install.packages("keras")
# install.packages("caTools")
library(keras)
library(caTools)

# Memuat dataset Dogecoin dan Shiba Inu
dogecoin_data <- read.csv("E:/homework/Crypto Project/dogecoin.csv")
shiba_inu_data <- read.csv("E:/homework/Crypto Project/shiba-inu.csv")

# Pra-pemrosesan Data
# Fokus pada 'price' sebagai target dan 'total_volume' sebagai fitur
dogecoin_data <- na.omit(dogecoin_data[, c("price", "total_volume")])
shiba_inu_data <- na.omit(shiba_inu_data[, c("price", "total_volume")])

# Menghapus baris yang memiliki nilai total_volume nol atau negatif
dogecoin_data <- dogecoin_data[dogecoin_data$total_volume > 0, ]
shiba_inu_data <- shiba_inu_data[shiba_inu_data$total_volume > 0, ]

# Normalisasi data (karena LSTM bekerja lebih baik dengan data yang sudah dinormalisasi)
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

dogecoin_data$price <- normalize(dogecoin_data$price)
dogecoin_data$total_volume <- normalize(dogecoin_data$total_volume)

shiba_inu_data$price <- normalize(shiba_inu_data$price)
shiba_inu_data$total_volume <- normalize(shiba_inu_data$total_volume)

# Membagi data menjadi set pelatihan dan uji (80% latih, 20% uji)
set.seed(123)
split_doge <- sample.split(dogecoin_data$price, SplitRatio = 0.8)
train_doge <- subset(dogecoin_data, split_doge == TRUE)
test_doge <- subset(dogecoin_data, split_doge == FALSE)

split_shiba <- sample.split(shiba_inu_data$price, SplitRatio = 0.8)
train_shiba <- subset(shiba_inu_data, split_shiba == TRUE)
test_shiba <- subset(shiba_inu_data, split_shiba == FALSE)

# Mempersiapkan data untuk LSTM (menggunakan 60 timesteps sebelumnya untuk memprediksi harga berikutnya)
create_sequences <- function(data, timesteps) {
  X <- NULL
  Y <- NULL
  for (i in seq(timesteps, nrow(data))) {
    X <- rbind(X, data[(i-timesteps+1):i, 2])  # Menggunakan total_volume sebagai fitur
    Y <- c(Y, data[i, 1])  # Menggunakan price sebagai target
  }
  return (list(X = array(X, dim = c(nrow(X), timesteps, 1)), Y = Y))
}

timesteps <- 60  # Menggunakan 60 timesteps untuk memprediksi harga berikutnya
train_doge_seq <- create_sequences(train_doge, timesteps)
test_doge_seq <- create_sequences(test_doge, timesteps)

train_shiba_seq <- create_sequences(train_shiba, timesteps)
test_shiba_seq <- create_sequences(test_shiba, timesteps)

# Membangun model LSTM
build_lstm_model <- function(input_shape) {
  model <- keras_model_sequential() %>%
    layer_lstm(units = 50, return_sequences = TRUE, input_shape = input_shape) %>%
    layer_lstm(units = 50, return_sequences = FALSE) %>%
    layer_dense(units = 25) %>%
    layer_dense(units = 1)
  
  model %>% compile(
    loss = "mean_squared_error",
    optimizer = "adam"
  )
  
  return(model)
}

# Melatih model untuk Dogecoin
model_doge <- build_lstm_model(input_shape = c(timesteps, 1))
model_doge %>% fit(train_doge_seq$X, train_doge_seq$Y, epochs = 10, batch_size = 32)

# Melatih model untuk Shiba Inu
model_shiba <- build_lstm_model(input_shape = c(timesteps, 1))
model_shiba %>% fit(train_shiba_seq$X, train_shiba_seq$Y, epochs = 10, batch_size = 32)

# Membuat prediksi pada data uji
pred_doge <- model_doge %>% predict(test_doge_seq$X)
pred_shiba <- model_shiba %>% predict(test_shiba_seq$X)

# Menghitung Mean Squared Error (MSE) untuk kedua model
mse_doge <- mean((test_doge_seq$Y - pred_doge)^2)
mse_shiba <- mean((test_shiba_seq$Y - pred_shiba)^2)

# Mencetak hasil
cat("Model Dogecoin - MSE:", mse_doge, "\n")
cat("Model Shiba Inu - MSE:", mse_shiba, "\n")

# Plot prediksi vs aktual untuk Dogecoin
plot(test_doge_seq$Y, type = "l", col = "blue", lwd = 2, 
     main = "Prediksi Harga Dogecoin vs Aktual (LSTM)", ylab = "Price", xlab = "Timesteps")
lines(pred_doge, col = "red", lwd = 2)
legend("topright", legend = c("Aktual", "Prediksi"), col = c("blue", "red"), lty = 1)

# Plot prediksi vs aktual untuk Shiba Inu
plot(test_shiba_seq$Y, type = "l", col = "blue", lwd = 2, 
     main = "Prediksi Harga Shiba Inu vs Aktual (LSTM)", ylab = "Price", xlab = "Timesteps")
lines(pred_shiba, col = "red", lwd = 2)
legend("topright", legend = c("Aktual", "Prediksi"), col = c("blue", "red"), lty = 1)
