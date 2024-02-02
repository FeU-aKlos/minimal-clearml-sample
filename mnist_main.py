from mnist_pipeline import create_workflow
from clearml.automation.controller import PipelineDecorator

def main():
    PipelineDecorator.set_default_execution_queue('k8s_scheduler')#
    
    train_images_url="http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"   
    train_labels_url = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
    test_images_url = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
    test_labels_url = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
    create_workflow(
        train_images_url=train_images_url,
        train_labels_url=train_labels_url,
        test_images_url=test_images_url,
        test_labels_url=test_labels_url
    )
    print("pipeline completed")
    
        
if __name__ == "__main__":
    main()