package eus.ehu.tss;


import ij.IJ;
import ij.ImagePlus;
import weka.classifiers.AggregateableEvaluation;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;

public class EvaluationTest {
    public static void main(final String[] args){

        ArrayList<RegionFeatures.Feature> selectedFeatures = new ArrayList<>();
        String[] selectedFs = RegionFeatures.Feature.getAllLabels();
        for(int i=0;i<selectedFs.length;++i){
            selectedFeatures.add(RegionFeatures.Feature.fromLabel(selectedFs[i]));
        }

        final ArrayList<String> classes = new ArrayList<String>();
        classes.add("background");
        classes.add("tumoral");
        classes.add("nontumoral");

        ImagePlus inImage = null;
        ImagePlus spImage = null;
        ImagePlus gtImage = null;
        AggregateableEvaluation totalEval = null;
        Instances training = null;
        Instances testing = null;

        RandomForest exampleClassifier = new RandomForest();

        ArrayList<Instances> dataSet = new ArrayList<>();
        System.out.println("Calculating image features and classes");
        for(int i=0;i<10;++i) {
            System.out.println("\tCalculating features of image "+String.format("%02d",i+1));
            inImage = IJ.openImage(EvaluationTest.class.getResource("/eval/histogram-matched-TMA/TMA-"+String.format("%02d",i+1)+".png").getFile());
            spImage = IJ.openImage(EvaluationTest.class.getResource("/eval/superpixels/SLIC-"+String.format("%02d",i+1)+".zip").getFile());
            gtImage = IJ.openImage(EvaluationTest.class.getResource("/eval/groundtruth/groundtruth-"+String.format("%02d",i+1)+".png").getFile());
            dataSet.add(RegionColorFeatures.calculateLabeledColorFeatures(inImage,spImage,gtImage,selectedFeatures,classes));
        }

        try {
            System.out.println("Merging data for fold 01");
            training = dataSet.get(1);
            System.out.print("\tAdding datasets to training data: 02");
            testing = dataSet.get(0);
            for(int i=2;i<10;++i){
                training=merge(training,dataSet.get(i));
                System.out.print(", "+String.format("%02d",i+1));
            }
            System.out.print("\n");
        }catch (Exception e){
            e.printStackTrace();
        }
        try {
            System.out.println("Training classifier");
            Evaluation eval = new Evaluation(training);
            exampleClassifier.buildClassifier(training);
            System.out.println("Used classifier\n\t"+exampleClassifier.toString());
            System.out.println("Evaluating model");
            eval.evaluateModel(exampleClassifier,testing);
            totalEval = new AggregateableEvaluation(eval);
            totalEval.aggregate(eval);
            System.out.println("Test result: ");
            System.out.println(eval.toSummaryString("\n\t1 results\n======\n",false));
            System.out.println(eval.toMatrixString());
        }catch (Exception e){
            e.printStackTrace();
        }
        try {
            System.out.println("Saving training file");
            BufferedWriter writer = new BufferedWriter(new FileWriter("trainingData1.arff"));
            writer.write(training.toString());
            writer.flush();
            writer.close();
        }catch (Exception e){
            e.printStackTrace();
        }
        try {
            System.out.println("\tSaving testing file");
            BufferedWriter writer = new BufferedWriter(new FileWriter("testingData1.arff"));
            writer.write(training.toString());
            writer.flush();
            writer.close();
        }catch (Exception e) {
            e.printStackTrace();
        }

        for(int i=0;i<10;++i){
            try {
                System.out.println("Merging data for fold "+String.format("%02d",i+1));
                training = dataSet.get(0);
                testing = dataSet.get(i);
                System.out.print("\tAdding datasets to training data: 01");
                for(int j=1;j<10;++j){
                    if(j!=i) {
                        System.out.print(", "+String.format("%02d",j+1));
                        training = merge(training, dataSet.get(j));
                    }
                }
                System.out.print("\n");
            }catch (Exception e){
                e.printStackTrace();
            }
            try {
                System.out.println("Training classifier");
                Evaluation eval = new Evaluation(training);
                exampleClassifier.buildClassifier(training);
                System.out.println("Used classifier\n\t"+exampleClassifier.toString());
                System.out.println("Evaluating model");
                eval.evaluateModel(exampleClassifier,testing);
                totalEval.aggregate(eval);
                System.out.println("Test result: ");
                System.out.println(eval.toSummaryString("\n\t"+String.format("%02d",i+1)+" results\n======\n",false));
                System.out.println(eval.toMatrixString());
            }catch (Exception e){
                e.printStackTrace();
            }
            try {
                System.out.println("Saving training file");
                BufferedWriter writer = new BufferedWriter(new FileWriter("trainingData"+String.format("%02d",i+1)+".arff"));
                writer.write(training.toString());
                writer.flush();
                writer.close();
            }catch (Exception e){
                e.printStackTrace();
            }
            try {
                System.out.println("\tSaving testing file");
                BufferedWriter writer = new BufferedWriter(new FileWriter("testingData"+String.format("%02d",i+1)+".arff"));
                writer.write(training.toString());
                writer.flush();
                writer.close();
            }catch (Exception e) {
                e.printStackTrace();
            }
        }

        try {
            System.out.println("\n===Aggregated evaluation results===\n");
            System.out.println(totalEval.toMatrixString());
            System.out.println(totalEval.toSummaryString());
        }catch (Exception e){
            e.printStackTrace();
        }


    }

    public static Instances merge(Instances data1, Instances data2)
            throws Exception
    {
        // Check where are the string attributes
        int asize = data1.numAttributes();
        boolean strings_pos[] = new boolean[asize];
        for(int i=0; i<asize; i++)
        {
            Attribute att = data1.attribute(i);
            strings_pos[i] = ((att.type() == Attribute.STRING) ||
                    (att.type() == Attribute.NOMINAL));
        }

        // Create a new dataset
        Instances dest = new Instances(data1);
        dest.setRelationName(data1.relationName() + "+" + data2.relationName());

        ConverterUtils.DataSource source = new ConverterUtils.DataSource(data2);
        Instances instances = source.getStructure();
        Instance instance = null;
        while (source.hasMoreElements(instances)) {
            instance = source.nextElement(instances);
            dest.add(instance);

            // Copy string attributes
            for(int i=0; i<asize; i++) {
                if(strings_pos[i]) {
                    dest.instance(dest.numInstances()-1)
                            .setValue(i,instance.stringValue(i));
                }
            }
        }

        return dest;
    }
}
