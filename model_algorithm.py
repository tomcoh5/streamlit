import torch
from torch import nn

class LSTM_cell_AI_SUMMER(torch.nn.Module):
    
#defining each unit

    def __init__(self, input_length=10, hidden_length=20):
        super(LSTM_cell_AI_SUMMER, self).__init__()
        self.input_length = input_length
        self.hidden_length = hidden_length

        # forget gate - what i dont use?
        self.linear_forget_w1 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_forget_r1 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_forget = nn.Sigmoid()

        # input gate 
        self.linear_gate_w2 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_gate_r2 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_gate = nn.Sigmoid()

        # cell memory components
        self.linear_gate_w3 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_gate_r3 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.activation_gate = nn.Tanh()

        # out gate - what do i pass to the next label?
        self.linear_gate_w4 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_gate_r4 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_hidden_out = nn.Sigmoid()

        self.activation_final = nn.Tanh()

    def forget(self, x, h):
        x = self.linear_forget_w1(x)
        h = self.linear_forget_r1(h)
        return self.sigmoid_forget(x + h)

    def input_gate(self, x, h):

        # Equation 1. input gate
        x_temp = self.linear_gate_w2(x)
        h_temp = self.linear_gate_r2(h)
        i = self.sigmoid_gate(x_temp + h_temp)
        return i

    def cell_memory_gate(self, i, f, x, h, c_prev):
        x = self.linear_gate_w3(x)
        h = self.linear_gate_r3(h)

        # new information part that will be injected in the new context
        k = self.activation_gate(x + h)
        g = k * i

        
        # forget old context/cell info
        c = f * c_prev
        # learn new context/cell info
        c_next = g + c
        return c_next

    def out_gate(self, x, h):
        x = self.linear_gate_w4(x)
        h = self.linear_gate_r4(h)
        return self.sigmoid_hidden_out(x + h)
    
    
    # scale the datasets - because the model is sensitive
    scaler = MinMaxScaler(feature_range=(0,1))

    dataset_train_scaled = scaler.fit_transform(dataset_train)
    dataset_test_scaled  = scaler.transform(dataset_test)

    
    #using the mini-max-scaler
    
    def mini-max-scaler(X,X.min,X.max,min,max) {
    X_std = (X - X.min() / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
             
    X_scaled = scale * X + min - X.min(axis=0) * scale
    where scale = (max - min) / (X.max(axis=0) - X.min(axis=0))
             }

      #feature - labels (creating the sets)
      def create_dataset(df):
         x = [] 
         y = [] 
 
      for i in range(30, df.shape[0]):
         #calling the mini-max-scaler
         mini-max-scaler(X,X.min,X.max,min,max)    
         x.append(df[i-30:i,0])
         y.append(df[i,0])
    
         x = np.array(x)
         y = np.array(y)
  
      for i in range(31, df.shape[1]):
         #calling the mini-max-scaler
         mini-max-scaler(X,X.min,X.max,min,max)    
         x1.append(df[i-31:i,1])
         y1.append(df[i,1])
    
         x1 = np.array(x)
         y1 = np.array(y)
             
          x = (x+x1)/2
          y = (x+x2)/2
             
     return x,y   
      
             #linear regression
             def estimate_coef(x, y):
    n = np.size(x)
    m_x = np.mean(x)
    m_y = np.mean(y)
 
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
 
    # calculating regression 
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
 
    return (b_0, b_1)
 
def plot_regression_line(x, y, b):
    x = np.array()
    y = np.array()
 
    # estimating 
    b = estimate_coef(x, y)
    b1 = (X.max + X.min)/2
    b2 = (Y.max + Y.min)/2
    
    b = (b+b1+b2)/3
             
    plot_regression_line(x, y, b)
             
     #building the new unit
    def forward(self, x, tuple_in ):
        (h, c_prev) = tuple_in
             
        #input gate
        i = self.input_gate(x*0.8 + b*0.2, h)

        #forget gate
        f = self.forget(x*0.8 + b*0.2, h)

        #updating the cell memory
        c_next = self.cell_memory_gate(i, f, x*0.8 + b*0.2, h,c_prev)

        #calculate the main output gate
        o = self.out_gate(x*0.8 + b*0.2, h)


        #produce next hidden output
        h_next = o * self.activation_final(c_next)

        
        return h_next, c_next
