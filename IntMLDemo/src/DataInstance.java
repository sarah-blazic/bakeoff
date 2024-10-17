import java.io.Serializable;
import java.util.Date;

/* A recorded instance of the training data */

public class DataInstance implements Serializable {

	private static final long serialVersionUID = -1L;
	public String label;
	public Date date;
	public float[] measurements;
	
	public String toCSVRow() {
		StringBuilder sb = new StringBuilder();
		sb.append(',');
		for (int i = 0; i < measurements.length; i++) {
			sb.append(measurements[i]);
			sb.append(',');
		}
		sb.append(label);
		return sb.toString();
	}
}