package eus.ehu.tss;


import ij.IJ;
import ij.ImagePlus;
import weka.classifiers.AggregateableEvaluation;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;

import java.io.*;
import java.util.ArrayList;
import java.util.zip.GZIPOutputStream;

public class EvaluationTest {
    public static void main(final String[] args) {
        IJ.log("****************Starting BayesNet evaluation*******************");
        long startTime = System.currentTimeMillis();

        String[] selectedFs = new String[8];
        selectedFs[0]="Mean";
        selectedFs[1]="Min";
        selectedFs[2]="Max";
        selectedFs[3]="Median";
        selectedFs[4]="NeighborsMean";
        selectedFs[5]="NeighborsMin";
        selectedFs[6]="NeighborsMax";
        selectedFs[7]="NeighborsMedian";

        ArrayList<RegionFeatures.Feature> selectedFeatures = new ArrayList();
        for(int i=0;i<selectedFs.length;++i){
            selectedFeatures.add(RegionFeatures.Feature.fromLabel(selectedFs[i]));
            //print(selectedFeatures.get(i));
        }

        BayesNet exampleClassifier = new BayesNet();

        File inputDir =  new File(args[0]);
        File gtDir =  new File(args[1]);
        File spDir =  new File(args[2]);
        File outputDir = new File(args[3]);

        File[] listOfFiles = inputDir.listFiles();
        File[] listOfFilesGt = gtDir.listFiles();
        File[] listOfFilesSp = spDir.listFiles();

        ArrayList<String> classes = new ArrayList();
        classes.add("background");
        classes.add("tumoral");
        classes.add("nontumoral");

        int[] classIndextoLabel = new int[3];
        classIndextoLabel[0] = 1;
        classIndextoLabel[1] = 2;
        classIndextoLabel[2] = 3;

        Instances training = null;
        Instances testing = null;

        ArrayList<Instances> dataSet = new ArrayList<>();

        try {
            IJ.log("Calculating image features and classes");
            for (int i = 0; i < 4; ++i) {
                ImagePlus inImage = IJ.openImage(listOfFiles[i].getCanonicalPath());
                //inImage.show();
                ImagePlus gtImage = IJ.openImage(listOfFilesGt[i].getCanonicalPath());
                //gtImage.show();
                ImagePlus spImage = IJ.openImage(listOfFilesSp[i].getCanonicalPath());
                //spImage.show();
                dataSet.add(RegionColorFeatures.calculateLabeledColorFeatures(inImage,spImage,gtImage,selectedFeatures,classes));
            }
        }catch (Exception e){
            e.printStackTrace();
        }
        try {
            System.out.println("Merging data for fold 01");
            training = dataSet.get(1);
            System.out.print("\tAdding datasets to training data: 02");
            for(int i=2;i<4;++i){
                training=merge(training,dataSet.get(i));
                System.out.print(", "+String.format("%02d",i+1));
            }
            System.out.println("Saving training file");
            BufferedWriter writer = new BufferedWriter(new FileWriter(outputDir.getPath() + File.separator+"training.arff"));
            writer.write(training.toString());
            writer.flush();
            writer.close();
        }catch (Exception e){
            e.printStackTrace();
        }

        try {
            testing = dataSet.get(0);
            System.out.println("Saving testing file");
            BufferedWriter writer = new BufferedWriter(new FileWriter(outputDir.getPath() + File.separator+"testing.arff"));
            writer.write(testing.toString());
            writer.flush();
            writer.close();
            System.out.println("Training classifier");
            exampleClassifier.buildClassifier(training);
            try {
                System.out.println("Saving classifier");
                File sFile = new File(outputDir.getPath() + File.separator+"classifier.model");
                OutputStream os = new FileOutputStream(sFile);
                if (sFile.getName().endsWith(".gz"))
                {
                    os = new GZIPOutputStream(os);
                }
                ObjectOutputStream objectOutputStream = new ObjectOutputStream(os);
                objectOutputStream.writeObject(exampleClassifier);
                Instances trainHeader = new Instances(training,0);
                objectOutputStream.writeObject(trainHeader);
                objectOutputStream.flush();
                objectOutputStream.close();
            }
            catch (Exception e)
            {
                IJ.error("Save Failed", "Error when saving classifier into a file");
            }
            TrainableSuperpixelSegmentation tss = new TrainableSuperpixelSegmentation();
            tss.setClassifier(exampleClassifier);
            tss.setClassifierTrained(true);
            tss.setSelectedFeatures(selectedFeatures);
            ImagePlus inImage = IJ.openImage(listOfFiles[0].getCanonicalPath());
            ImagePlus spImage = IJ.openImage(listOfFilesSp[0].getCanonicalPath());
            tss.setInputImage(inImage);
            tss.setLabelImage(spImage);
            tss.setUnlabeled(testing);
            System.out.println("Evaluating model");
            ImagePlus result = tss.applyClassifier();
            result.show();
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
