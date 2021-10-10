import tensorflow as tf
import tensorflow.keras


def test_model(x_test, y_test, trained_model, save_pred=False, trust_indicator=True):
  
    test_score = trained_model.evaluate(x_test, y_test, verbose=0)
    
    print("")
    print('*** Test loss:    ', test_score[0])
    print('*** Test mean_iou:', test_score[1])
    print("")

    if trust_indicator:
        output_pred = trained_model.predict(x_test)
        output_pred.sort(axis=1)
        
        trust_diff = [(output_pred[i][-1] - output_pred[i][-2]) for i in range(x_test.shape[0])]
        trust_diff = np.array(trust_diff)
        
        trust_mean = trust_diff.mean(axis=0)
        trust_min  = np.amin(trust_diff)
        
        print("")
        print("***** Trust difference mean:", trust_mean)
        print("***** Minimal difference   :", trust_min)
        print("")


if __name__=="__main":
    x_test, y_test = 

    trained_model = keras.load_model('trained_models/test')
    
    test_model(x_test, y_test, trained_model)