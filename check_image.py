# Imports python modules
from time import time
# Imports functions created for this program
from get_input_args import get_input_args
from train import train, save_checkpoint, load_checkpoint
from predict import predict, show_result
from preprocess_data import preprocess_data, cat_to_names

train_data, valid_data, test_data, trainloader, validloader, testloader = preprocess_data()

# Main program function defined below
def main():
    start_time = time()
    input_args = get_input_args()
    cat_to_name = cat_to_names()
    image_path = input_args.image_path
    topk = input_args.top_k
    
    #trains a network and shows running_loss, test accuracy, valid loss and valid accuracy:
    train()
    
    #saves checkpoint:
    save_checkpoint()
    
    #gets model and class_to_idx from checkpoint:
    model, model.class_to_idx = load_checkpoint()
                
    #gets predicted name, max probability, topk probabilities and topk classes for test image:
    pred_name, pred_prob, top_probabilities, top_classes = predict(model)
    
    #Printing results:
    test_name = cat_to_name[image_path.split("/")[-2]]
    print("Pred_Name: {}, Probability: {}\n Test_Name: {}\n Top {} probabilities: {}\n Top {} classes: {}".format(pred_name, pred_prob, test_name, topk, top_probabilities, topk, top_classes))
    
    #Computes overall runtime in seconds & prints it in hh:mm:ss format:    
    end_time = time()
    tot_time = end_time - start_time 
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(round((tot_time%3600)%60)) )
    
    #Shows predicted image, flower name and plot top probabilities to top classes:
    show_result(top_probabilities, top_classes)    
    
# Call to main function to run the program
if __name__ == "__main__":
    main()
