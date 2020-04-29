
#libaray
pacman::p_load(tidyverse, keras, magick,jpeg,tensorflow, ggplot2, ggpubr)
install_keras()
theme_set(theme_pubr())
options(digits = 3)


list.files("input/imgs")

list.files('input/train')

head(list.files('input/imgs/test'))

head(read.csv('input/sample_submission.csv'))

driver_imgs_df<-read.csv('input/driver_imgs_list.csv')

driver_imgs_df<-read.csv('input/driver_imgs_list.csv')

dim(driver_imgs_df)

#plot class distrubution
ggplot(driver_imgs_df, aes(classname)) +
  geom_bar(fill = "#0073C2FF") +
  theme_pubclean()

img <- readJPEG("input/imgs/test/img_1.jpg") 

dim(img)

# image size to scale down to (original images are 480 x 640 px)
img_width <- 256
img_height <- 256
target_size <- c(img_width, img_height)

# RGB = 3 channels
channels <- 3

output_n =10

#train & test directorys
#path of image folders

train_dir <- 'input/imgs/train'
test_dir <- 'input/imgs/test'

#load images from directory and perform scaling

train_datagen = image_data_generator(rescale = 1/255, data_format='channels_last', validation_split=.5)
test_datagen <- image_data_generator(rescale = 1/255, data_format='channels_last')

train_generator <- flow_images_from_directory(
  train_dir,                  # Target directory  
  train_datagen,              # Data generator
  target_size = c(256, 256),  # Resizes all images
  batch_size = 32,
  class_mode = "categorical",
  seed = 89,
  subset='training'
)

validation_generator <- flow_images_from_directory(
  train_dir,                  # Target directory  
  train_datagen,              # Data generator
  target_size = c(256, 256),  # Resizes all images
  batch_size = 32,
  class_mode = "categorical",       
  subset='validation'
)

test_generator <- flow_images_from_directory(
  test_dir,                  # Target directory  
  test_datagen,              # Data generator
  target_size = c(256, 256),  # Resizes all images
  batch_size = 32,
  shuffle=FALSE,
  class_mode = NULL,
  seed = 89
)



#number of images per class

cat("Number of images per class:")
table(factor(train_generator$classes))


#class label vs index mapping

cat("\nClass label vs index mapping:\n")
train_generator$class_indices


classes_indices <- train_generator$class_indices
save(classes_indices, file = "classes_indices.RData")


#Define Model

# number of training samples
train_samples <- train_generator$n
# number of validation samples
test_samples <- test_generator$n

# define batch size and number of epochs
epochs <- 10

#Specify Model
model <- keras_model_sequential()

model %>%
  
  # Start with hidden 2D convolutional layer being fed 256x256 pixel images
  layer_conv_2d(
    filter = 16, kernel_size = c(3,3), padding = "same", 
    input_shape = c(img_width, img_height, channels),
      activation='relu') %>%
  layer_batch_normalization() %>%
  # Second hidden layer
  layer_conv_2d(filter = 128, kernel_size = c(3,3), activation='relu') %>%
  layer_batch_normalization() %>%    

  # Use max pooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate=.2) %>%
      
  # 2 additional hidden 2D convolutional layers
  layer_conv_2d(filter = 64, kernel_size = c(4,4), padding = "same", activation='relu') %>%
  layer_batch_normalization() %>%
  layer_conv_2d(filter = 64, kernel_size = c(4,4), activation='relu') %>%
  layer_batch_normalization() %>%

  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate=.2) %>%

  # Flatten max filtered output into feature vector 
  layer_flatten() %>%

  # Outputs from dense layer are projected onto 10 unit output layer
  layer_dense(output_n, activation='softmax')


#explore Model

str(model)



#Compile Model

opt <- optimizer_rmsprop(lr = 0.0001, decay = 1e-6)
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = opt,
  metrics = "accuracy"
)


#Fit Model

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 500,
  epochs = epochs,
  validation_data=validation_generator,
  validation_steps=100
)

plot(history)


#explore model accuracy

model %>% evaluate_generator(train_generator, steps=50)

#Validation accuracy
model %>% evaluate_generator(validation_generator, steps=50)

#Scoring Random Example

classes <- c('normal driving', 'texting - right', 'talking on the phone - right', 'texting - left',
             'talking on the phone - left', 'operating the radio', 'drinking', 'reaching behind',
             'hair and makeup', 'talking to passenger')

file <- sample(list.files(test_dir),1)

img_path <- paste(test_dir, file, sep='/')         
img <- image_load(img_path, target_size = c(256, 256)) %>%           
  image_to_array() %>%                                               
  array_reshape(dim = c(1, 256, 256, 3)) 


p_class <- predict_classes(model, img)
preds <- as.data.frame(t(predict_proba(model, img)))
preds$class <- classes


classes[p_class+1]
preds


img_ <- image_load(img_path, target_size = c(640, 480)) %>%           
  image_to_array() %>%                                               
  array_reshape(dim = c(1, 640, 480, 3)) 


img_tensor=img_/255
plot(as.raster(img_tensor[1,,,]))


#save the model

save_model_hdf5(model, 'model_v1.h5', overwrite = TRUE,
  include_optimizer = TRUE)

