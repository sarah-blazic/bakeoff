import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import weka.classifiers.functions.SMO;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

/* A wrapper class to use Weka's classifiers */

public class MLClassifier implements Serializable {
    // Mark featureCalc as transient if it's not serializable
    transient FeatureCalc featureCalc = null;
    SMO classifier = null;
    Attribute classattr;
    Filter filter = new Normalize();
    List<String> classNames; // Add this field to store class names

    public MLClassifier() {

    }

    public void train(Map<String, List<DataInstance>> instances) {

        /* Generate instances using the collected map of DataInstances */

        /* Store class labels */
        this.classNames = new ArrayList<>(instances.keySet());
        featureCalc = new FeatureCalc(this.classNames);

        /* Collect training data */
        List<DataInstance> trainingData = new ArrayList<>();

        for (List<DataInstance> v : instances.values()) {
            trainingData.addAll(v);
        }

        /* Prepare the training dataset */
        Instances dataset = featureCalc.calcFeatures(trainingData);

        /* Build the classifier */
        classifier = new SMO();

        try {
            // You can adjust the classifier options as needed
            classifier.setOptions(weka.core.Utils.splitOptions("-C 1.0 -L 0.0010 "
                    + "-P 1.0E-12 -N 0 -V -1 -W 1 "
                    + "-K \"weka.classifiers.functions.supportVector.PolyKernel "
                    + "-C 0 -E 1.0\""));

            classifier.buildClassifier(dataset);
            this.classattr = dataset.classAttribute();

            System.out.println("Training done!");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public String classify(DataInstance data) {
        if (classifier == null || classattr == null) {
            return "Unknown";
        }

        Instance instance = featureCalc.calcFeatures(data);

        try {
            int result = (int) classifier.classifyInstance(instance);
            return classattr.value(result);
        } catch (Exception e) {
            e.printStackTrace();
            return "Error";
        }
    }

    // Method to re-initialize transient fields after deserialization
    private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        // Re-initialize featureCalc using the stored classNames
        featureCalc = new FeatureCalc(this.classNames);
    }
}
