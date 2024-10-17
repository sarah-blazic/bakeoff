import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/* A helper class to calculate quantities in ML features */

public class FeatureCalc{
	
	Instances dataset;	
	List<String> classLabels;
	int nfeatures;
	boolean isFirstInstance = true;

	public FeatureCalc(List<String> classLabels) {
		this.classLabels = classLabels;
	}
	
	private Instance instanceFromArray(double[] valueArray, String label) {
		Instance instance = new DenseInstance(1.0, valueArray);

		instance.setDataset(dataset);
		if(label != null) {
			instance.setClassValue(label);
		} else {
			instance.setClassMissing();
		}

		return instance;
	}

	private Instance calcFirstInstance(DataInstance data) {
		final ArrayList<Attribute> attrs = new ArrayList<>();
		final ArrayList<Double> values = new ArrayList<>();
		
		//Yang: consider adding more features to make your demo more accurate and robust to ambient noise

		nfeatures = data.measurements.length;
		
		for(int i = 0; i < nfeatures; i++){
			attrs.add(new Attribute("bin"+i, i));
			values.add((double) data.measurements[i]);
		}

		/* build our dataset (instance header) */
		attrs.add(new Attribute("classlabel", classLabels, nfeatures));
		dataset = new Instances("dataset", attrs, 0);
		dataset.setClassIndex(nfeatures);
		
		/* build the output instance */
		double[] valueArray = new double[nfeatures+1];
		for(int i=0; i<nfeatures; i++) {
			valueArray[i] = values.get(i);
		}
		
		return instanceFromArray(valueArray, data.label);
		
	}

	private Instance calcOtherInstance(DataInstance data) {
		final double[] valueArray = new double[nfeatures+1];

		for(int i = 0; i < nfeatures; i++){
			valueArray[i] = data.measurements[i];
		}
		
		return instanceFromArray(valueArray, data.label);
	}

	public Instance calcFeatures(DataInstance data) {
		if(isFirstInstance) {
			isFirstInstance = false;
			return calcFirstInstance(data);
			
		} else {
			return calcOtherInstance(data);
		}
	}

	public Instances calcFeatures(Collection<DataInstance> dataCollection) {
		Instances res = null;
		for(DataInstance data : dataCollection) {
			Instance inst = calcFeatures(data);
			
			if(res == null) {
				res = new Instances(dataset, dataCollection.size());
			}
			res.add(inst);
		}
		return res;
	}
}