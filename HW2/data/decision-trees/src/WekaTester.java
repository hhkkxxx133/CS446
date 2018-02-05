package cs446.homework2;

import java.io.File;
import java.io.FileReader;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.Attribute;
import weka.core.FastVector;
import cs446.weka.classifiers.trees.Id3;
import java.util.Random;
import weka.core.converters.ArffSaver;

public class WekaTester {

    public static void main(String[] args) throws Exception {

	// if (args.length != 2) { /////////////////
	//     System.err.println("Usage: WekaTester arff-file");
	//     System.exit(-1);
	// }

	// // Load the data
	// Instances data = new Instances(new FileReader(new File(args[0])));
	// Instances test = new Instances(new FileReader(new File(args[1]))); //////////////

	// // The last attribute is the class label
	// data.setClassIndex(data.numAttributes() - 1);
	// test.setClassIndex(data.numAttributes() - 1); /////////////

	// // Train on 80% of the data and test on 20%
	// // Instances train = data.trainCV(5,0);
	// // Instances test = data.testCV(5, 0);

	// // Create a new ID3 classifier. This is the modified one where you can
	// // set the depth of the tree.
	// Id3 classifier = new Id3();

	// // An example depth. If this value is -1, then the tree is grown to full
	// // depth.
	// classifier.setMaxDepth(-1);

	// // Train
	// classifier.buildClassifier(data);

	// // Print the classfier
	// System.out.println(classifier);
	// System.out.println();

	// // Evaluate on the test set
	// Evaluation evaluation = new Evaluation(test);
	// evaluation.evaluateModel(classifier, test);
	// System.out.println(evaluation.toSummaryString());


		// Load the data into 5 folds
		Instances trains[]= new Instances[5];
		for (int i = 0; i < 5; ++i){
			trains[i] = new Instances(new FileReader(new File(args[i])));
			trains[i].setClassIndex(trains[i].numAttributes() - 1);
		}
		Instances tests[] = new Instances[5];
		for (int i = 0; i < 5; ++i){
			tests[i] = new Instances(new FileReader(new File(args[i+5])));
			tests[i].setClassIndex(tests[i].numAttributes() - 1);
		}

		// 5 Id3 with full depth
		Id3 id3_full[] = new Id3[5];
		Evaluation id3_full_eval[]= new Evaluation[5];
		for (int i = 0; i < 5; i ++){
			id3_full[i] = new Id3();
			id3_full[i].setMaxDepth(-1);
			id3_full[i].buildClassifier(trains[i]);
			id3_full_eval[i] = new Evaluation(tests[i]);
			id3_full_eval[i].evaluateModel(id3_full[i],tests[i]);
			System.out.println("ID3 with full depth, fold " + (i + 1) + "\n");
			System.out.println(id3_full[i]);
			System.out.println();
			System.out.println(id3_full_eval[i].toSummaryString());
		}
		
		// 5 Id3 with depth of 4
		Id3 id3_d4[] = new Id3[5];
		Evaluation id3_d4_eval[]= new Evaluation[5];
		for (int i = 0; i < 5; i ++){
			id3_d4[i] = new Id3();
			id3_d4[i].setMaxDepth(4);
			id3_d4[i].buildClassifier(trains[i]);
			id3_d4_eval[i] = new Evaluation(tests[i]);
			id3_d4_eval[i].evaluateModel(id3_d4[i],tests[i]);
			System.out.println("ID3 with depth of 4, fold " + (i + 1) + "\n");
			System.out.println(id3_d4[i]);
			System.out.println();
			System.out.println(id3_d4_eval[i].toSummaryString());
		}

		// 5 Id3 with depth of 8
		Id3 id3_d8[] = new Id3[5];
		Evaluation id3_d8_eval[]= new Evaluation[5];
		for (int i = 0; i < 5; i ++){
			id3_d8[i] = new Id3();
			id3_d8[i].setMaxDepth(8);
			id3_d8[i].buildClassifier(trains[i]);
			id3_d8_eval[i] = new Evaluation(tests[i]);
			id3_d8_eval[i].evaluateModel(id3_d8[i],tests[i]);
			System.out.println("ID3 with depth of 8, fold " + (i + 1) + "\n");
			System.out.println(id3_d8[i]);
			System.out.println();
			System.out.println(id3_d8_eval[i].toSummaryString());
		}

		Id3 stumps[][] = new Id3[5][100];
		for(int i = 0; i < 5; i ++){
			for(int j = 0; j < 100; j ++){
				stumps[i][j] = new Id3();
				stumps[i][j].setMaxDepth(4);
				Instances subset = new Instances(trains[i]);
				Random rand = new Random(); 
				int value = rand.nextInt(200); 
				subset.randomize(new Random(value));
				for(int k = 0; k < trains[i].numInstances()/2; k++){
					subset.delete(k);
				}
				stumps[i][j].buildClassifier(subset);
			}
		}
		Instances new_trains[] = new Instances[5];
		Instances new_tests[] = new Instances[5];
		for(int i = 0; i < 5; i ++){
			FastVector atts_train = new FastVector(101);
			FastVector atts_test = new FastVector(101);
			for(int j = 0; j < 100; j ++){
				atts_train.addElement(new Attribute(Integer.toString(j)));
				atts_test.addElement(new Attribute(Integer.toString(j)));
			}
			atts_train.addElement(new Attribute("label", (FastVector) null));
			atts_test.addElement(new Attribute("label", (FastVector) null));
			new_trains[i] = new Instances("fold" + i, atts_train, 70);
			new_tests[i] = new Instances("fold" + i, atts_test, 70);
			for(int j = 0; j < trains[i].numInstances(); j ++){
				Instance cur = new Instance(101);
				cur.setDataset(new_trains[i]);
				for(int k = 0; k < 100; k ++){
					cur.setValue(k,stumps[i][k].classifyInstance(trains[i].instance(j)));
				}
				cur.setValue(100, trains[i].instance(j).stringValue(trains[i].instance(j).numAttributes() - 1));
				new_trains[i].add(cur);
			}
			for(int j = 0; j < tests[i].numInstances(); j ++){
				Instance cur2 = new Instance(101);
				cur2.setDataset(new_tests[i]);
				for(int k = 0; k < 100; k++){
					cur2.setValue(k,stumps[i][k].classifyInstance(tests[i].instance(j)));
				}
				cur2.setValue(100,tests[i].instance(j).stringValue(tests[i].instance(j).numAttributes()-1));
				new_tests[i].add(cur2);
			}
			new_trains[i].setClassIndex(new_trains[i].numAttributes() - 1);
			new_tests[i].setClassIndex(new_tests[i].numAttributes()-1);
		}
		ArffSaver saver = new ArffSaver();
		for(int i = 0; i < 5; i ++){
			saver.setInstances(new_trains[i]);
			saver.setFile(new File("new_trains" + (i + 1) + ".arff"));
			saver.writeBatch();
			saver.setInstances(new_tests[i]);
			saver.setFile(new File("new_test" + (i + 1) + ".arff"));
			saver.writeBatch();
		}
    }
}
