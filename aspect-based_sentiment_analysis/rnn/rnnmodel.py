import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义LSTM模型
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,label_num,dropout,bidirectional,model_name):
        super(RNNModel, self).__init__()
        if model_name == "lstm":
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,bidirectional=bidirectional)
        elif model_name == "gru":
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,bidirectional=bidirectional)
        else:
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True,bidirectional=bidirectional)

        self.model_name = model_name
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.l1 = nn.Linear(hidden_size*2, 128)
        else:
            self.l1 = nn.Linear(hidden_size, 128)
        self.drop1 = nn.Dropout(dropout)

        self.l2 = nn.Linear(128, 64)
        self.drop2 = nn.Dropout(dropout)
        self.l3 = nn.Linear(64, 32)
        self.drop3 = nn.Dropout(dropout)

        self.l4 = nn.Linear(32, 16)
        self.drop4 = nn.Dropout(dropout)



        self.fc_0 = nn.Linear(16, label_num)
        self.fc_1 = nn.Linear(16, label_num)
        self.fc_2 = nn.Linear(16, label_num)
        self.fc_3 = nn.Linear(16, label_num)
        self.fc_4 = nn.Linear(16, label_num)
        self.fc_5 = nn.Linear(16, label_num)
        self.fc_6 = nn.Linear(16, label_num)
        self.fc_7 = nn.Linear(16, label_num)
        self.fc_8 = nn.Linear(16, label_num)
        self.fc_9 = nn.Linear(16, label_num)
        self.fc_10 = nn.Linear(16, label_num)

        self.drop = nn.Dropout(dropout)

    def forward(self, x, lengths):
        x_packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # RNN forward

        if self.model_name == "lstm":
            packed_output, (hidden, _) = self.rnn(x_packed)
        else:
            packed_output, hidden = self.rnn(x_packed)


        # Unpack the sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # We use the hidden state of the last layer as the representation
        if self.bidirectional:
            hidden_last_layer = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden_last_layer = hidden[-1, :, :]

        output = self.l1(hidden_last_layer)
        output = self.drop1(output)
        output = self.l2(output)
        output = self.drop2(output)
        output = self.l3(output)
        output = self.drop3(output)

        output = self.l4(output)
        output = self.drop4(output)



        out0 = F.log_softmax(self.fc_0(output), dim=1)
        out1 = F.log_softmax(self.fc_1(output), dim=1)
        out2 = F.log_softmax(self.fc_2(output), dim=1)
        out3 = F.log_softmax(self.fc_3(output), dim=1)
        out4 = F.log_softmax(self.fc_4(output), dim=1)
        out5 = F.log_softmax(self.fc_5(output), dim=1)
        out6 = F.log_softmax(self.fc_6(output), dim=1)
        out7 = F.log_softmax(self.fc_7(output), dim=1)
        out8 = F.log_softmax(self.fc_8(output), dim=1)
        out9 = F.log_softmax(self.fc_9(output), dim=1)
        out10 = F.log_softmax(self.fc_10(output), dim=1)

        out = torch.stack((out0, out1, out2, out3, out4, out5, out6, out7, out8, out9, out10), dim=1)



        return out
