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
        classes.add("blue");
        classes.add("red");

        ImagePlus inImage = null;
        ImagePlus spImage = null;
        ImagePlus gtImage = null;

        RandomForest exampleClassifier = new RandomForest();
        System.out.println("Calculating image features and classes");
        System.out.println("\tCalculating features of image 1");
        inImage = IJ.openImage(EvaluationTest.class.getResource("/eval/histogram-matched-TMA/TMA-01.png").getFile());
        spImage = IJ.openImage(EvaluationTest.class.getResource("/eval/superpixels/SLIC-01.zip").getFile());
        gtImage = IJ.openImage(EvaluationTest.class.getResource("/eval/groundtruth/groundtruth-01.png").getFile());
        Instances data1 = RegionColorFeatures.calculateLabeledColorFeatures(inImage,spImage,gtImage,selectedFeatures,classes);
        System.out.println("\tCalculating features of image 2");
        inImage = IJ.openImage(EvaluationTest.class.getResource("/eval/histogram-matched-TMA/TMA-02.png").getFile());
        spImage = IJ.openImage(EvaluationTest.class.getResource("/eval/superpixels/SLIC-02.zip").getFile());
        gtImage = IJ.openImage(EvaluationTest.class.getResource("/eval/groundtruth/groundtruth-02.png").getFile());
        Instances data2 = RegionColorFeatures.calculateLabeledColorFeatures(inImage,spImage,gtImage,selectedFeatures,classes);
        System.out.println("\tCalculating features of image 3");
        inImage = IJ.openImage(EvaluationTest.class.getResource("/eval/histogram-matched-TMA/TMA-03.png").getFile());
        spImage = IJ.openImage(EvaluationTest.class.getResource("/eval/superpixels/SLIC-03.zip").getFile());
        gtImage = IJ.openImage(EvaluationTest.class.getResource("/eval/groundtruth/groundtruth-03.png").getFile());
        Instances data3 = RegionColorFeatures.calculateLabeledColorFeatures(inImage,spImage,gtImage,selectedFeatures,classes);
        System.out.println("\tCalculating features of image 4");
        inImage = IJ.openImage(EvaluationTest.class.getResource("/eval/histogram-matched-TMA/TMA-04.png").getFile());
        spImage = IJ.openImage(EvaluationTest.class.getResource("/eval/superpixels/SLIC-04.zip").getFile());
        gtImage = IJ.openImage(EvaluationTest.class.getResource("/eval/groundtruth/groundtruth-04.png").getFile());
        Instances data4 = RegionColorFeatures.calculateLabeledColorFeatures(inImage,spImage,gtImage,selectedFeatures,classes);
        System.out.println("\tCalculating features of image 5");
        inImage = IJ.openImage(EvaluationTest.class.getResource("/eval/histogram-matched-TMA/TMA-05.png").getFile());
        spImage = IJ.openImage(EvaluationTest.class.getResource("/eval/superpixels/SLIC-05.zip").getFile());
        gtImage = IJ.openImage(EvaluationTest.class.getResource("/eval/groundtruth/groundtruth-05.png").getFile());
        Instances data5 = RegionColorFeatures.calculateLabeledColorFeatures(inImage,spImage,gtImage,selectedFeatures,classes);
        System.out.println("\tCalculating features of image 6");
        inImage = IJ.openImage(EvaluationTest.class.getResource("/eval/histogram-matched-TMA/TMA-06.png").getFile());
        spImage = IJ.openImage(EvaluationTest.class.getResource("/eval/superpixels/SLIC-06.zip").getFile());
        gtImage = IJ.openImage(EvaluationTest.class.getResource("/eval/groundtruth/groundtruth-06.png").getFile());
        Instances data6 = RegionColorFeatures.calculateLabeledColorFeatures(inImage,spImage,gtImage,selectedFeatures,classes);
        System.out.println("\tCalculating features of image 7");
        inImage = IJ.openImage(EvaluationTest.class.getResource("/eval/histogram-matched-TMA/TMA-07.png").getFile());
        spImage = IJ.openImage(EvaluationTest.class.getResource("/eval/superpixels/SLIC-07.zip").getFile());
        gtImage = IJ.openImage(EvaluationTest.class.getResource("/eval/groundtruth/groundtruth-07.png").getFile());
        Instances data7 = RegionColorFeatures.calculateLabeledColorFeatures(inImage,spImage,gtImage,selectedFeatures,classes);
        System.out.println("\tCalculating features of image 8");
        inImage = IJ.openImage(EvaluationTest.class.getResource("/eval/histogram-matched-TMA/TMA-08.png").getFile());
        spImage = IJ.openImage(EvaluationTest.class.getResource("/eval/superpixels/SLIC-08.zip").getFile());
        gtImage = IJ.openImage(EvaluationTest.class.getResource("/eval/groundtruth/groundtruth-08.png").getFile());
        Instances data8 = RegionColorFeatures.calculateLabeledColorFeatures(inImage,spImage,gtImage,selectedFeatures,classes);
        System.out.println("\tCalculating features of image 9");
        inImage = IJ.openImage(EvaluationTest.class.getResource("/eval/histogram-matched-TMA/TMA-09.png").getFile());
        spImage = IJ.openImage(EvaluationTest.class.getResource("/eval/superpixels/SLIC-09.zip").getFile());
        gtImage = IJ.openImage(EvaluationTest.class.getResource("/eval/groundtruth/groundtruth-09.png").getFile());
        Instances data9 = RegionColorFeatures.calculateLabeledColorFeatures(inImage,spImage,gtImage,selectedFeatures,classes);
        System.out.println("\tCalculating features of image 10");
        inImage = IJ.openImage(EvaluationTest.class.getResource("/eval/histogram-matched-TMA/TMA-10.png").getFile());
        spImage = IJ.openImage(EvaluationTest.class.getResource("/eval/superpixels/SLIC-10.zip").getFile());
        gtImage = IJ.openImage(EvaluationTest.class.getResource("/eval/groundtruth/groundtruth-10.png").getFile());
        Instances data10 = RegionColorFeatures.calculateLabeledColorFeatures(inImage,spImage,gtImage,selectedFeatures,classes);
        inImage = null;
        spImage = null;
        gtImage = null;
        System.gc();

        AggregateableEvaluation totalEval = null;
        Instances training = null;
        Instances testing = null;
        try {
            System.out.println("Merging data for fold 1");
            training = merge(data2, data3);
            training = merge(training,data4);
            training = merge(training,data5);
            training = merge(training,data6);
            training = merge(training,data7);
            training = merge(training,data8);
            training = merge(training,data9);
            training = merge(training,data10);
            testing = data1;
        }catch (Exception e){
            e.printStackTrace();
        }
        try {
            System.out.println("Training classifier");
            Evaluation eval = new Evaluation(training);
            exampleClassifier.buildClassifier(training);
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

        try {
            System.out.println("Merging data for fold 2");
            training = merge(data1, data3);
            training = merge(training,data4);
            training = merge(training,data5);
            training = merge(training,data6);
            training = merge(training,data7);
            training = merge(training,data8);
            training = merge(training,data9);
            training = merge(training,data10);
            testing = data2;
        }catch (Exception e){
            e.printStackTrace();
        }
        try {
            System.out.println("Training classifier");
            Evaluation eval = new Evaluation(training);
            exampleClassifier.buildClassifier(training);
            System.out.println("Evaluating model");
            eval.evaluateModel(exampleClassifier,testing);
            totalEval.aggregate(eval);
            System.out.println("Test result: ");
            System.out.println(eval.toSummaryString("\n\t2 results\n======\n",false));
            System.out.println(eval.toMatrixString());
        }catch (Exception e){
            e.printStackTrace();
        }
        try {
            System.out.println("Saving training file");
            BufferedWriter writer = new BufferedWriter(new FileWriter("trainingData2.arff"));
            writer.write(training.toString());
            writer.flush();
            writer.close();
        }catch (Exception e){
            e.printStackTrace();
        }
        try {
            System.out.println("\tSaving testing file");
            BufferedWriter writer = new BufferedWriter(new FileWriter("testingData2.arff"));
            writer.write(training.toString());
            writer.flush();
            writer.close();
        }catch (Exception e) {
            e.printStackTrace();
        }

        try {
            System.out.println("Merging data for fold 3");
            training = merge(data1, data2);
            training = merge(training,data4);
            training = merge(training,data5);
            training = merge(training,data6);
            training = merge(training,data7);
            training = merge(training,data8);
            training = merge(training,data9);
            training = merge(training,data10);
            testing = data3;
        }catch (Exception e){
            e.printStackTrace();
        }
        try {
            System.out.println("Training classifier");
            Evaluation eval = new Evaluation(training);
            exampleClassifier.buildClassifier(training);
            System.out.println("Evaluating model");
            eval.evaluateModel(exampleClassifier,testing);
            totalEval.aggregate(eval);
            System.out.println("Test result: ");
            System.out.println(eval.toSummaryString("\n\t 3 results\n======\n",false));
            System.out.println(eval.toMatrixString());
        }catch (Exception e){
            e.printStackTrace();
        }
        try {
            System.out.println("Saving training file");
            BufferedWriter writer = new BufferedWriter(new FileWriter("trainingData3.arff"));
            writer.write(training.toString());
            writer.flush();
            writer.close();
        }catch (Exception e){
            e.printStackTrace();
        }
        try {
            System.out.println("\tSaving testing file");
            BufferedWriter writer = new BufferedWriter(new FileWriter("testingData3.arff"));
            writer.write(training.toString());
            writer.flush();
            writer.close();
        }catch (Exception e) {
            e.printStackTrace();
        }


        try {
            System.out.println("Merging data for fold 4");
            training = merge(data1, data2);
            training = merge(training,data3);
            training = merge(training,data5);
            training = merge(training,data6);
            training = merge(training,data7);
            training = merge(training,data8);
            training = merge(training,data9);
            training = merge(training,data10);
            testing = data4;
        }catch (Exception e){
            e.printStackTrace();
        }
        try {
            System.out.println("Training classifier");
            Evaluation eval = new Evaluation(training);
            exampleClassifier.buildClassifier(training);
            System.out.println("Evaluating model");
            eval.evaluateModel(exampleClassifier,testing);
            totalEval.aggregate(eval);
            System.out.println("Test result: ");
            System.out.println(eval.toSummaryString("\n\t 4 results\n======\n",false));
        }catch (Exception e){
            e.printStackTrace();
        }
        try {
            System.out.println("Saving training file");
            BufferedWriter writer = new BufferedWriter(new FileWriter("trainingData4.arff"));
            writer.write(training.toString());
            writer.flush();
            writer.close();
        }catch (Exception e){
            e.printStackTrace();
        }
        try {
            System.out.println("\tSaving testing file");
            BufferedWriter writer = new BufferedWriter(new FileWriter("testingData4.arff"));
            writer.write(training.toString());
            writer.flush();
            writer.close();
        }catch (Exception e) {
            e.printStackTrace();
        }

        try {
            System.out.println("Merging data for fold 5");
            training = merge(data1, data2);
            training = merge(training,data3);
            training = merge(training,data4);
            training = merge(training,data6);
            training = merge(training,data7);
            training = merge(training,data8);
            training = merge(training,data9);
            training = merge(training,data10);
            testing = data5;
        }catch (Exception e){
            e.printStackTrace();
        }
        try {
            System.out.println("Training classifier");
            Evaluation eval = new Evaluation(training);
            exampleClassifier.buildClassifier(training);
            System.out.println("Evaluating model");
            eval.evaluateModel(exampleClassifier,testing);
            totalEval.aggregate(eval);
            System.out.println("Test result: ");
            System.out.println(eval.toSummaryString("\n\t 5 results\n======\n",false));
        }catch (Exception e){
            e.printStackTrace();
        }
        try {
            System.out.println("Saving training file");
            BufferedWriter writer = new BufferedWriter(new FileWriter("trainingData5.arff"));
            writer.write(training.toString());
            writer.flush();
            writer.close();
        }catch (Exception e){
            e.printStackTrace();
        }
        try {
            System.out.println("\tSaving testing file");
            BufferedWriter writer = new BufferedWriter(new FileWriter("testingData5.arff"));
            writer.write(training.toString());
            writer.flush();
            writer.close();
        }catch (Exception e) {
            e.printStackTrace();
        }


        try {
            System.out.println("Merging data for fold 6");
            training = merge(data1, data2);
            training = merge(training,data3);
            training = merge(training,data4);
            training = merge(training,data5);
            training = merge(training,data7);
            training = merge(training,data8);
            training = merge(training,data9);
            training = merge(training,data10);
            testing = data6;
        }catch (Exception e){
            e.printStackTrace();
        }
        try {
            System.out.println("Training classifier");
            Evaluation eval = new Evaluation(training);
            exampleClassifier.buildClassifier(training);
            System.out.println("Evaluating model");
            eval.evaluateModel(exampleClassifier,testing);
            totalEval.aggregate(eval);
            System.out.println("Test result: ");
            System.out.println(eval.toSummaryString("\n\t 6 results\n======\n",false));
        }catch (Exception e){
            e.printStackTrace();
        }
        try {
            System.out.println("Saving training file");
            BufferedWriter writer = new BufferedWriter(new FileWriter("trainingData6.arff"));
            writer.write(training.toString());
            writer.flush();
            writer.close();
        }catch (Exception e){
            e.printStackTrace();
        }
        try {
            System.out.println("\tSaving testing file");
            BufferedWriter writer = new BufferedWriter(new FileWriter("testingData6.arff"));
            writer.write(training.toString());
            writer.flush();
            writer.close();
        }catch (Exception e) {
            e.printStackTrace();
        }


        try {
            System.out.println("Merging data for fold 7");
            training = merge(data1, data2);
            training = merge(training,data3);
            training = merge(training,data4);
            training = merge(training,data5);
            training = merge(training,data6);
            training = merge(training,data8);
            training = merge(training,data9);
            training = merge(training,data10);
            testing = data7;
        }catch (Exception e){
            e.printStackTrace();
        }
        try {
            System.out.println("Training classifier");
            Evaluation eval = new Evaluation(training);
            exampleClassifier.buildClassifier(training);
            System.out.println("Evaluating model");
            eval.evaluateModel(exampleClassifier,testing);
            totalEval.aggregate(eval);
            System.out.println("Test result: ");
            System.out.println(eval.toSummaryString("\n\t 7 results\n======\n",false));
        }catch (Exception e){
            e.printStackTrace();
        }
        try {
            System.out.println("Saving training file");
            BufferedWriter writer = new BufferedWriter(new FileWriter("trainingData7.arff"));
            writer.write(training.toString());
            writer.flush();
            writer.close();
        }catch (Exception e){
            e.printStackTrace();
        }
        try {
            System.out.println("\tSaving testing file");
            BufferedWriter writer = new BufferedWriter(new FileWriter("testingData7.arff"));
            writer.write(training.toString());
            writer.flush();
            writer.close();
        }catch (Exception e) {
            e.printStackTrace();
        }

        try {
            System.out.println("Merging data for fold 8");
            training = merge(data1, data2);
            training = merge(training,data3);
            training = merge(training,data4);
            training = merge(training,data5);
            training = merge(training,data6);
            training = merge(training,data7);
            training = merge(training,data9);
            training = merge(training,data10);
            testing = data8;
        }catch (Exception e){
            e.printStackTrace();
        }
        try {
            System.out.println("Training classifier");
            Evaluation eval = new Evaluation(training);
            exampleClassifier.buildClassifier(training);
            System.out.println("Evaluating model");
            eval.evaluateModel(exampleClassifier,testing);
            totalEval.aggregate(eval);
            System.out.println("Test result: ");
            System.out.println(eval.toSummaryString("\n\t 8 results\n======\n",false));
        }catch (Exception e){
            e.printStackTrace();
        }
        try {
            System.out.println("Saving training file");
            BufferedWriter writer = new BufferedWriter(new FileWriter("trainingData8.arff"));
            writer.write(training.toString());
            writer.flush();
            writer.close();
        }catch (Exception e){
            e.printStackTrace();
        }
        try {
            System.out.println("\tSaving testing file");
            BufferedWriter writer = new BufferedWriter(new FileWriter("testingData8.arff"));
            writer.write(training.toString());
            writer.flush();
            writer.close();
        }catch (Exception e) {
            e.printStackTrace();
        }


        try {
            System.out.println("Merging data for fold 9");
            training = merge(data1, data2);
            training = merge(training,data3);
            training = merge(training,data4);
            training = merge(training,data5);
            training = merge(training,data6);
            training = merge(training,data7);
            training = merge(training,data8);
            training = merge(training,data10);
            testing = data9;
        }catch (Exception e){
            e.printStackTrace();
        }
        try {
            System.out.println("Training classifier");
            Evaluation eval = new Evaluation(training);
            exampleClassifier.buildClassifier(training);
            System.out.println("Evaluating model");
            eval.evaluateModel(exampleClassifier,testing);
            totalEval.aggregate(eval);
            System.out.println("Test result: ");
            System.out.println(eval.toSummaryString("\n\t 9 results\n======\n",false));
        }catch (Exception e){
            e.printStackTrace();
        }
        try {
            System.out.println("Saving training file");
            BufferedWriter writer = new BufferedWriter(new FileWriter("trainingData9.arff"));
            writer.write(training.toString());
            writer.flush();
            writer.close();
        }catch (Exception e){
            e.printStackTrace();
        }
        try {
            System.out.println("\tSaving testing file");
            BufferedWriter writer = new BufferedWriter(new FileWriter("testingData9.arff"));
            writer.write(training.toString());
            writer.flush();
            writer.close();
        }catch (Exception e) {
            e.printStackTrace();
        }

        try {
            System.out.println("Merging data for fold 10");
            training = merge(data1, data2);
            training = merge(training,data3);
            training = merge(training,data4);
            training = merge(training,data5);
            training = merge(training,data6);
            training = merge(training,data7);
            training = merge(training,data8);
            training = merge(training,data9);
            testing = data10;
        }catch (Exception e){
            e.printStackTrace();
        }
        try {
            System.out.println("Training classifier");
            Evaluation eval = new Evaluation(training);
            exampleClassifier.buildClassifier(training);
            System.out.println("Evaluating model");
            eval.evaluateModel(exampleClassifier,testing);
            totalEval.aggregate(eval);
            System.out.println("Test result: ");
            System.out.println(eval.toSummaryString("\n\t 10 results\n======\n",false));
        }catch (Exception e){
            e.printStackTrace();
        }
        try {
            System.out.println("Saving training file");
            BufferedWriter writer = new BufferedWriter(new FileWriter("trainingData10.arff"));
            writer.write(training.toString());
            writer.flush();
            writer.close();
        }catch (Exception e){
            e.printStackTrace();
        }
        try {
            System.out.println("\tSaving testing file");
            BufferedWriter writer = new BufferedWriter(new FileWriter("testingData10.arff"));
            writer.write(training.toString());
            writer.flush();
            writer.close();
        }catch (Exception e) {
            e.printStackTrace();
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
