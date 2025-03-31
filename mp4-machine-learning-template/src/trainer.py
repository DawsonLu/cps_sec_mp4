import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from data_reader import DataReader


class Trainer:
    """
    Class responsible for training a neural network model.
    """
    def __init__(self, model: nn.Module) -> None:
        """
        Initializes the Trainer with a model and an optional learning rate.
        
        Parameters:
            model (nn.Module): The neural network model to be trained.
        """
        self.model = model
        
        # Define the loss function. 
        self.criterion = nn.BCELoss()  # Define the appropriate loss function
        
        # Define the optimizer.
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=0.001,
        )  # Initialize the optimizer with model parameters and learning rate

    def train_model(self, data_reader: DataReader) -> None:
        """
        Trains the model on data provided by the DataReader instance.
        
        Parameters:
            data_reader: An instance of DataReader containing the training data and labels.
        Returns:
            None
        """
        # Create DataLoader for mini-batch processing
        train_dataset = TensorDataset(data_reader.X_tensor, data_reader.y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
        epochs: int = 100 # Defien the number of epochs to train the model for
    
        # Training loop
        for epoch in range(epochs):
            # Iterate over batches of data
            total_loss = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):  # Use your DataLoader here
                # Reset gradients via zero_grad()
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(data)
                
                # Compute loss
                loss = self.criterion(outputs, target)
                
                # Backward pass and optimize via backward() and optimizer.step()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
            # You can print the loss here to see how it decreases
            