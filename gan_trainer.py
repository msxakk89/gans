import sys
import os
import argparse
import contextlib
import numpy as np

# Disable warnings/debugging info printing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Dense, Reshape, Flatten
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.optimizers import Adam
import simplejson # for saving model architecture

def build_generator():
    model = Sequential()
    model.add(Dense(256, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))
    return model

def build_discriminator():
    model = Sequential()
    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model


def build_gan(g,d):
    model = Sequential()
    # Fix discriminator for GAN training
    d.trainable = False
    model.add(g)
    model.add(d)
    model.compile(loss='binary_crossentropy', optimizer=adam)
    return model

# Global variables for plotting
import matplotlib.pyplot as plt
import glob
import imageio
import PIL
save_name = 0.00000000

def save_imgs(r,c,path):
    noise = np.random.normal(0, 1, (r * c, latent_dim))
    gen_imgs = generator.predict(noise)
    global save_name
    save_name += 0.00000001

    # Scale images
    gen_imgs = 127.5 * gen_imgs + 1.

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("%s/%.8f.png" % (path,save_name))
    plt.close()

def save_model_evolution(epochs,y,label,save_path):
    e = list(range(epochs))
    fig, axs = plt.subplots(1, 1)
    axs.plot(e,y)
    axs.title.set_text(label)
    plt.xlabel("Epoch")
    path = save_path
    fname = label
    n = "%s/%s.png" % (path,fname)
    fig.savefig(n)
    plt.close()
    
def train(epochs, batch_size, save_interval,X_train):
    # Rescale images
    X_train = X_train / 127.5 - 1.

    valid = np.ones((batch_size, 1))
    fakes = np.zeros((batch_size, 1))
    lst_d_loss = []
    lst_d_acc = []
    lst_g_loss = []
    
    for epoch in range(epochs):
        # Get Random Batch
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]

        # Generate Fake Images
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_imgs = generator.predict(noise)

        # Train discriminator
        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fakes)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, latent_dim))

        # Train GAN
        g_loss = GAN.train_on_batch(noise, valid)

        acc = 100* d_loss[1]

        lst_d_loss.append(d_loss[0])
        lst_d_acc.append(acc)
        lst_g_loss.append(g_loss)

        if(epoch % save_interval) == 0:
            print("******* %d [D loss: %f, acc: %.2f%%] [G loss: %f]" %
                  (epoch, d_loss[0], acc, g_loss))
            save_imgs(r=args.r,c=args.c,path=args.path)
        
    save_model_evolution(epochs=epochs,
                         y=lst_d_loss,
                         label="disc_loss",
                         save_path=args.path)
    save_model_evolution(epochs=epochs,
                         y=lst_d_acc,
                         label="disc_acc",
                         save_path=args.path)
    save_model_evolution(epochs=epochs,
                         y=lst_g_loss,
                         label="gen_loss",
                         save_path=args.path)

def create_animation(anim_file):
    if ".gif" not in anim_file:
        anim_file += ".gif"
    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob('%s/*.png' % args.path)
        filenames.remove('disc_loss.png')
        filenames.remove('disc_acc.png')
        filenames.remove('gen_loss.png')
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)

def save_models(path,d,g):
    # serialize model architecture to JSON
    g_json = g.to_json()
    with open(path + "/generator.json", "w") as json_file:
        json_file.write(simplejson.dumps(simplejson.loads(g_json), indent=4))
    d_json = d.to_json()
    with open(path + "/discriminator.json", "w") as json_file:
        json_file.write(simplejson.dumps(simplejson.loads(d_json), indent=4))
    
    # serialize model weights to HDF5
    g.save_weights(path + "/generator.h5")
    d.save_weights(path + "/discriminator.h5") 

if __name__ == '__main__':
    '''
    Command line options
    '''
    parser = argparse.ArgumentParser(
        description='Train Generative Adversarial Network (GAN) using Keras datasets')
    parser.add_argument('--d', dest='d',
                        type=str,
                        default='mnist',
                        help="Select Keras pictures data to train GAN. Choose from 'mnist' and 'fashion_mnist'. Defulats to 'mnist' else 'fashion_mnist' chosen")
    parser.add_argument('--epochs', dest='epochs',
                        type=int,
                        default=5000,
                        help="Define number of epochs for GAN training. Defaults to 5000")
    parser.add_argument('--batch_size', dest='batch',
                        type=int,
                        default=64,
                        help="Define batch size. Defaults to 64")
    parser.add_argument('--interval', dest='interval',
                        type=int,
                        default=200,
                        help="Define interval after which generated images are drawn and training stats are displayed. Defaults to 200")
    parser.add_argument('--r', dest='r',
                        type=int,
                        default=5,
                        help="Number of rows in generated image. Defaults to 5")
    parser.add_argument('--c', dest='c',
                        type=int,
                        default=5,
                        help="Number of columns in generated image. Defaults to 5")
    parser.add_argument('--path', dest='path',
                        type=str,
                        default="images",
                        help="Folder name or full path where drawn images will be saved. Defaults to 'images'")
    parser.add_argument('--mod_path', dest='mod_path',
                        type=str,
                        default="models",
                        help="Folder name or full path where models will be saved. Defaults to 'models'")
    parser.add_argument('--train_on_number', dest='train_on_number',
                        type=int,
                        default=-1,
                        help="If 'mnist' data selected train on a subset of this number only. Must be an integer between 0 and 9. Defaults to -1 which implies all numbers")
    parser.add_argument('--train_on_subset', dest='train_on_subset',
                        type=int,
                        default=-1,
                        help="If 'fashion_mnist' data selected train on a subset of this number only. Must be an integer between 0 and 9. See [https://keras.io/api/datasets/fashion_mnist/] for the explanation of numbers. Defaults to -1 which implies all types")
    parser.add_argument('--create_anim', dest='create_anim',
                        type=bool,
                        default=False,
                        help="Should a GIF animation be created? Defaults to False")
    parser.add_argument('--anim_file', dest='anim_file',
                        type=str,
                        default="anim.gif",
                        help="Name of GIF animation file. Should end with .gif. Defaults to 'anim.gif'")

    args = parser.parse_args()

    print("Creating folder to store images")
    try:
        os.mkdir(args.path)
    except:
        print("Supplied images folder name already exists. Please use another one")
        sys.exit()
    
    print("Creating folder to store models")
    try:
        os.mkdir(args.mod_path)
    except:
        print("Supplied model folder name already exists. Please use another one")
        sys.exit()
        
    print("Loading data")
    if args.d == "mnist":
        from keras.datasets import mnist as data
        if args.train_on_number == -1:
            (X_train, _), (_, _) = data.load_data()
            num_pics = X_train.shape[0]
        else:
            (X_train, y_train), (_, _) = data.load_data()
            b = y_train==args.train_on_number
            X_train = X_train[b]
            num_pics = X_train.shape[0]
    else:
        from keras.datasets import fashion_mnist as data
        if args.train_on_subset == - 1:
            (X_train, _), (_, _) = data.load_data()
            num_pics = X_train.shape[0]
        else:
            (X_train, y_train), (_, _) = data.load_data()
            b = y_train==args.train_on_subset
            X_train = X_train[b]
            num_pics = X_train.shape[0]
        
    print("You have loaded %d images from %s" %(num_pics,args.d))

    print("Setting global variables")
    img_width = 28
    img_height = 28
    channels = 1
    img_shape = (img_width, img_height, channels)
    latent_dim = 100
    adam = Adam(learning_rate=0.0001)

    print("Building generator")
    generator = build_generator()

    print("Building discriminator")
    discriminator = build_discriminator()

    print("Building GAN")
    GAN = build_gan(g=generator,d=discriminator)

    print("Starting GAN training")
    train(args.epochs,args.batch,args.interval,X_train)

    if args.create_anim:
        print("Making GIF")
        create_animation(args.anim_file)
    
    print("Saving models")
    save_models(path=args.mod_path,d=discriminator,g=generator)
    
    print("DONE")
