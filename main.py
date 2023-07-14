import pandas as pd
import numpy as np
import argparse
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50
from transformers import BertModel, BertTokenizer

# 清空缓存的GPU内存
import torch.cuda
torch.cuda.empty_cache()

def load_img(path, index):
    imgs_path = []
    for i in index:
        image_path = path + str(i) + ".jpg"
        # image = cv.imread(image_path)
        imgs_path.append(image_path)
    return imgs_path

def load_txt(path, index):
    txts = []
    max_len = 0
    for i in index:
        txt_path = path + str(i) + ".txt"
        with open(txt_path, "r", encoding="GB18030") as f:
            txt = f.read().split()
            washed_wds = ''
            for word in txt:
                if not word.startswith(("@", "#", "|", "http")):
                    washed_wds.join(word)
            if len(washed_wds) > max_len:
                max_len = len(washed_wds)
            txts.append(washed_wds)
    return txts, max_len

class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, tokenized_texts, labels, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.input_ids = [x['input_ids'] for x in tokenized_texts]
        self.attention_mask = [x['attention_mask'] for x in tokenized_texts]
        self.labels = labels

    def __getitem__(self, index):
        input_ids = torch.tensor(self.input_ids[index])
        attention_mask = torch.tensor(self.attention_mask[index])
        labels = torch.tensor(self.labels[index])
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        image = self.transform(image)

        return image, input_ids, attention_mask, labels

    def __len__(self):
        return len(self.input_ids)

class ImageExtractor(nn.Module):
    def __init__(self):
        super(ImageExtractor, self).__init__()
        self.resnet = resnet50(pretrained=True)

    def forward(self, image):
        features = self.resnet(image)
        return features

class TextExtractor(nn.Module):
    def __init__(self):
        super(TextExtractor, self).__init__()
        self.bert = pretrained_model

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        output = pooled_output
        return output

class FusionModel(nn.Module):
    def __init__(self, num_classes, option):
        super(FusionModel, self).__init__()
        self.image_extractor = ImageExtractor()
        self.text_encoder = TextExtractor()
        self.option = option
        # 仅输入图像特征
        self.img_only = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
            nn.ReLU(inplace=True),
        )
        # 仅输入文本特征
        self.txt_only = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
            nn.ReLU(inplace=True),
        )

        # 图像+文本
        self.img_txt = nn.Sequential(
            nn.Linear(1768, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes),
            nn.ReLU(inplace=True),
        )

    def forward(self, image, input_ids, attention_mask):
        if self.option == 0:
            image_features = self.image_extractor(image)
            output = image_features
            output = self.img_only(image_features)
        elif self.option == 1:
            text_features = self.text_encoder(input_ids, attention_mask)
            output = self.txt_only(text_features)
        else:
            image_features = self.image_extractor(image)
            text_features = self.text_encoder(input_ids, attention_mask)
            fusion_features = torch.cat((text_features, image_features), dim=-1)
            output = self.img_txt(fusion_features)
        return output

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    total_correct = 0
    for images, input_ids, attention_mask, labels in train_loader:
        images = images.to(device)
        input_ids = input_ids.squeeze(1).to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images, input_ids, attention_mask)
        _, preds = torch.max(outputs, 1)
        total_correct += torch.sum(preds == labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = total_correct.item() / len(train_loader.dataset)
    return epoch_loss, epoch_acc

def predict(model, test_loader, device):
    model.eval()
    predictions = []
    for images, input_ids, attention_mask, _ in test_loader:
        images = images.to(device)
        input_ids = input_ids.squeeze(1).to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            outputs = model(images, input_ids, attention_mask)
            _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())
    return predictions

if __name__ == '__main__':

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    pretrained_model = BertModel.from_pretrained("bert-base-multilingual-cased")

    parser = argparse.ArgumentParser()
    parser.add_argument('--option', type=int, default=2, help='0-img_only 1-txt_only 2-img_txt')
    args = parser.parse_args()

    train_data = pd.read_csv("train.txt")
    train_data = train_data.replace({"tag": {"positive": 0, "negative": 1, "neutral": 2}})
    index = list(train_data["guid"])
    train_label = list(train_data["tag"])

    pred_data = pd.read_csv("test_without_label.txt")
    pred_index = list(pred_data["guid"])
    pred_label = np.zeros(len(pred_data))

    path = "./data/"
    train_img = load_img(path, index)
    train_txt, max_len = load_txt(path, index)
    pred_img = load_img(path, pred_index)
    pred_txt, pred_max_len = load_txt(path, pred_index)

    train_txt = [tokenizer(text, padding='max_length', max_length=max_len, truncation=True, return_tensors="pt") for text in train_txt]
    pred_txt = [tokenizer(text, padding='max_length', max_length=pred_max_len, truncation=True, return_tensors="pt") for text in pred_txt]

    train_img, test_image, train_txt, test_txt, train_label, test_label = \
        train_test_split(train_img, train_txt, train_label, test_size=0.2, random_state=2023)
    # print(train_img[0], train_txt[0], train_label[0])

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    train_dataset = Dataset(train_img, train_txt, train_label, transform)
    test_dataset = Dataset(test_image, test_txt, test_label, transform)
    pred_dataset = Dataset(pred_img, pred_txt, pred_label, transform)

    lr, batch_size, epoch_nums, best_acc = 0.001, 32, 10, 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(0)
    criterion = nn.CrossEntropyLoss()
    model = FusionModel(3, args.option).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    pred_iter = DataLoader(pred_dataset, batch_size=batch_size, shuffle=True)

    print("train start")
    for epoch in range(epoch_nums):
        train_loss, train_acc = train(model, train_iter, criterion, optimizer, device)
        test_pred = predict(model, test_iter, device)
        test_acc = (np.array(test_pred) == np.array(test_label)).sum() / len(test_label)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model, 'multi_model.pt')
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {test_acc:.4f}, Best Acc:{best_acc:.4f}")
    print("train finish")

    print("predict start")
    best_model = torch.load('multi_model.pt').to(device)
    prediction = predict(best_model, test_iter, device)
    prediction = np.array(prediction)
    result = pd.read_csv('test.csv')
    result.loc[:, 'tag'] = prediction
    if args.option == 0:
        result.to_csv('img_only_result.csv', index=0)
    elif args.option == 1:
        result.to_csv('txt_only_result.csv', index=0)
    else:
        result.to_csv('img_txt_result.csv', index=0)
    print("predict finish")
