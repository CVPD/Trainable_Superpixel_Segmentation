/*
 * To the extent possible under law, the ImageJ developers have waived
 * all copyright and related or neighboring rights to this tutorial code.
 *
 * See the CC0 1.0 Universal license for details:
 *     http://creativecommons.org/publicdomain/zero/1.0/
 */

//Have just changed the name of the file

package eus.ehu.tss;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.WindowManager;
import ij.gui.*;
import ij.plugin.PlugIn;
import ij.process.ImageProcessor;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.lazy.IBk;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;


public class Trainable_Superpixel_Segmentation implements PlugIn {

    private CustomWindow win;
    private ImagePlus inputImage;
    private ImagePlus supImage;
    private ImagePlus resultImage;
    private final ExecutorService exec = Executors.newFixedThreadPool(1);
    private int numClasses = 2;
    private  java.awt.List[] exampleList = new java.awt.List[numClasses];
    private ArrayList<int[]> tags = new ArrayList<>();
    private ArrayList<RegionFeatures.Feature> features;
    private AbstractClassifier classifier;
    private ArrayList<String> classes;
    private TrainableSuperpixelSegmentation trainableSuperpixelSegmentation;

    private class CustomWindow extends StackWindow
    {
        private Panel all = new Panel();
        private JPanel controlsPanel = new JPanel();
        private JPanel classifierPanel = new JPanel();
        private JPanel resultPanel = new JPanel();
        private JPanel annotationsPanel = new JPanel();
        private JButton trainClassButton = null;
        private JButton loadClassButton = null;
        private JButton applyClassButton = null;
        private JButton settButton = null;
        private JButton plotButton = null;
        private JButton probButton = null;
        private JButton resButton = null;
        private JButton overlayButton = null;
        private JButton [] addExampleButton = new JButton[numClasses];


        private ActionListener listener = new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                final String command = e.getActionCommand();

                exec.submit(new Runnable() {
                    @Override
                    public void run() {
                        if(e.getSource() == trainClassButton)
                        {
                            runStopTraining(command);
                        }
                        else if(e.getSource() == overlayButton){
                            toggleOverlay();
                        }
                        else if(e.getSource() == resButton){
                           createResult();
                        }
                        else if(e.getSource() == probButton){
                            showProbability();
                        }
                        else if(e.getSource() == plotButton){
                            showPlot();
                        }
                        else if(e.getSource() == applyClassButton){
                            applyClassifier();
                        }
                        else if(e.getSource() == loadClassButton){
                            loadClassifier();
                        }
                        else if(e.getSource() == settButton){
                            showSettingsDialog();
                        }
                        else{
                            for(int i = 0; i < numClasses; i++)
                            {
                                if(e.getSource() == exampleList[i])
                                {
                                    deleteSelected(e);
                                    break;
                                }
                                if(e.getSource() == addExampleButton[i])
                                {
                                    addExamples(i);
                                    break;
                                }
                            }
                        }
                    }
                });
            }
        };

        private ItemListener itemListener = new ItemListener() {
            @Override
            public void itemStateChanged(ItemEvent e) {
                exec.submit(new Runnable() {
                    @Override
                    public void run() {
                        for(int i = 0; i < numClasses; i++)
                        {
                            if(e.getSource() == exampleList[i])
                                listSelected(e, i);
                        }
                    }
                });
            }
        };

        CustomWindow(ImagePlus imp)
        {
            super(imp, new ImageCanvas(imp));
            final ImageCanvas canvas = (ImageCanvas) getCanvas();
            GridBagLayout layout = new GridBagLayout();
            GridBagConstraints allConstraints = new GridBagConstraints();
            all.setLayout(layout);
            allConstraints.anchor = GridBagConstraints.CENTER;
            allConstraints.fill = GridBagConstraints.VERTICAL;
            allConstraints.gridwidth = 1;
            allConstraints.gridheight = 1;
            allConstraints.gridx = 0;
            allConstraints.gridy = 0;
            allConstraints.weightx = 0;
            allConstraints.weighty = 0;
            all.add(canvas,allConstraints);
            allConstraints.gridy++;

            //Control panel layout and constraints
            GridBagLayout controlLayout = new GridBagLayout();
            controlsPanel.setLayout(controlLayout);
            GridBagConstraints controlConstraints = new GridBagConstraints();
            controlConstraints.anchor = GridBagConstraints.CENTER;
            controlConstraints.fill = GridBagConstraints.HORIZONTAL;
            controlConstraints.gridwidth = 1;
            controlConstraints.gridheight = 1;
            controlConstraints.gridx = 0;
            controlConstraints.gridy = 0;
            controlConstraints.weightx = 0;
            controlConstraints.weighty = 0;

            //Classifier panel layout and constraints
            GridBagLayout classifierLayout = new GridBagLayout();
            classifierPanel.setLayout(classifierLayout);
            GridBagConstraints classifierConstraints = new GridBagConstraints();
            classifierConstraints.anchor = GridBagConstraints.WEST;
            classifierConstraints.fill = GridBagConstraints.VERTICAL;
            classifierConstraints.gridwidth = 1;
            classifierConstraints.gridheight = 1;
            classifierConstraints.gridx = 0;
            classifierConstraints.gridy = 0;
            classifierConstraints.weightx = 0;
            classifierConstraints.weighty = 0;

            //Result panel layout and constraints
            GridBagLayout resultLayout = new GridBagLayout();
            resultPanel.setLayout(resultLayout);
            GridBagConstraints resultConstraints = new GridBagConstraints();
            resultConstraints.anchor = GridBagConstraints.WEST;
            resultConstraints.fill = GridBagConstraints.VERTICAL;
            resultConstraints.gridwidth = 1;
            resultConstraints.gridheight = 1;
            resultConstraints.gridx = 0;
            resultConstraints.gridy = 0;
            resultConstraints.weightx = 0;
            resultConstraints.weighty = 0;


            // Annotations panel
            GridBagLayout boxAnnotation = new GridBagLayout();
            annotationsPanel.setBorder(BorderFactory.createTitledBorder("Labels"));
            annotationsPanel.setLayout(boxAnnotation);
            GridBagConstraints annotationsConstraints = new GridBagConstraints();
            annotationsConstraints.anchor = GridBagConstraints.NORTHWEST;
            annotationsConstraints.fill = GridBagConstraints.HORIZONTAL;
            annotationsConstraints.gridwidth = 1;
            annotationsConstraints.gridheight = 1;
            annotationsConstraints.gridx = 0;
            annotationsConstraints.gridy = 0;

            //Classifier panel buttons
            trainClassButton = new JButton("Train classifier");
            classifierPanel.add(trainClassButton,classifierConstraints);
            classifierConstraints.gridy++;
            loadClassButton = new JButton("Load classifier");
            classifierPanel.add(loadClassButton,classifierConstraints);
            classifierConstraints.gridy++;
            applyClassButton = new JButton("Apply classifier");
            classifierPanel.add(applyClassButton,classifierConstraints);
            classifierConstraints.gridy++;
            settButton = new JButton("Settings");
            classifierPanel.add(settButton,classifierConstraints);
            classifierConstraints.gridy++;
            controlsPanel.add(classifierPanel,controlConstraints);
            controlConstraints.gridx++;

            //Result panel buttons
            plotButton = new JButton("Plot data");
            resultPanel.add(plotButton,resultConstraints);
            resultConstraints.gridy++;
            probButton = new JButton("Get probability");
            resultPanel.add(probButton,resultConstraints);
            resultConstraints.gridy++;
            resButton = new JButton("Create result");
            resultPanel.add(resButton,resultConstraints);
            resultConstraints.gridy++;
            overlayButton = new JButton("Toggle overlay");
            resultPanel.add(overlayButton,resultConstraints);
            resultConstraints.gridy++;
            controlsPanel.add(resultPanel,controlConstraints);
            controlConstraints.gridx++;

            //Annotations panel
            for(int i=0; i<numClasses;++i){
                exampleList[i].addActionListener(listener);
                exampleList[i].addItemListener(itemListener);
                addExampleButton[i] = new JButton("Add to class "+i);
                addExampleButton[i].setToolTipText("Add markings of label 'class "+i+"'");

                annotationsConstraints.insets = new Insets(5,5,6,6);

                annotationsPanel.add(addExampleButton[i],annotationsConstraints);
                annotationsConstraints.gridy++;

                annotationsConstraints.insets = new Insets(0,0,0,0);

                annotationsPanel.add(exampleList[i], annotationsConstraints);
                annotationsConstraints.gridy++;
            }
            addExampleButton[0].setSelected(true);

            //Add listeners
            for(int i = 0; i< numClasses; ++i){
                addExampleButton[i].addActionListener(listener);
            }
            trainClassButton.addActionListener(listener);
            loadClassButton.addActionListener(listener);
            applyClassButton.addActionListener(listener);
            settButton.addActionListener(listener);
            plotButton.addActionListener(listener);
            probButton.addActionListener(listener);
            resButton.addActionListener(listener);
            overlayButton.addActionListener(listener);

            all.add(controlsPanel,allConstraints);
            allConstraints.gridx++;
            all.add(annotationsPanel,allConstraints);

            GridBagLayout wingb = new GridBagLayout();
            GridBagConstraints winc = new GridBagConstraints();
            winc.anchor = GridBagConstraints.CENTER;
            winc.fill = GridBagConstraints.BOTH;
            winc.weightx = 0;
            winc.weighty = 0;
            setLayout( wingb );
            add( all, winc );
            pack();
        }
    }


    void runStopTraining(final String command){
        if(trainableSuperpixelSegmentation.trainClassifier(tags)){
            resultImage = trainableSuperpixelSegmentation.applyClassifier();
            resultImage.show();
        }
    }

    void applyClassifier(){
        System.out.println("To be implemented: Apply classifier");
    }

    void loadClassifier(){
        System.out.println("To be implemented: Load classifier");
    }

    void showSettingsDialog(){
        System.out.println("To be implemented: Show settings dialog");
    }

    void showPlot(){
        System.out.println("To be implemented: Show Plot");
    }

    void showProbability(){
        System.out.println("To be implemented: Show probability");
    }

    void createResult(){
        System.out.println("To be implemented: Create result");
    }

    void toggleOverlay(){
        System.out.println("To be implemented: Toggle overlay");
    }

    void deleteSelected(final ActionEvent e){
        System.out.println("To be implemented: Delete selected "+e.toString());
    }

    /**
     * Adds tags based on the ROIs selected by the user
     * @param i identifier of class to add tags
     */
    void addExamples(int i){

        final Roi r = inputImage.getRoi();
        if(null == r){
            System.out.println("ROI null");
            return;
        }
        ArrayList<Float> selectedLabel = getSelectedLabels(supImage,r);
        for(Float label: selectedLabel){
            inputImage.killRoi();
            int[] a = tags.get(i);
            a = Arrays.copyOf(a, a.length+1);
            a[a.length-1] = label.intValue();
            tags.set(i,a);
            exampleList[i].add(label.toString());
        }
    }

    void listSelected(final ItemEvent e, final int i)
    {
       System.out.println("To be implemented: List selected: e: "+e.toString()+" i: "+i);
    }

    /**
     * Taken from MorphoLibJ, LabelImages class
     *
     * Get list of selected labels in label image. Labels are selected by
     * either a freehand ROI or point ROIs. Zero-value label is skipped.
     *
     * @param labelImage  label image
     * @param roi  FreehandRoi or PointRoi with selected labels
     * @return list of selected labels
     */
    public static ArrayList<Float> getSelectedLabels(
            final ImagePlus labelImage,
            final Roi roi )
    {
        final ArrayList<Float> list = new ArrayList<Float>();

        // if the user makes point selections
        if( roi instanceof PointRoi)
        {
            int[] xpoints = roi.getPolygon().xpoints;
            int[] ypoints = roi.getPolygon().ypoints;

            // read label values at those positions
            if( labelImage.getImageStackSize() > 1 )
            {
                final ImageStack labelStack = labelImage.getImageStack();
                for ( int i = 0; i<xpoints.length; i ++ )
                {
                    float value = (float) labelStack.getVoxel(
                            xpoints[ i ],
                            ypoints[ i ],
                            ((PointRoi) roi).getPointPosition( i )-1 );
                    if( Float.compare( 0f, value ) != 0 &&
                            list.contains( value ) == false )
                        list.add( (float) value );
                }
            }
            else
            {
                final ImageProcessor ip = labelImage.getProcessor();
                for ( int i = 0; i<xpoints.length; i ++ )
                {
                    float value = ip.getf( xpoints[ i ], ypoints[ i ]);
                    if( Float.compare( 0f, value ) != 0 &&
                            list.contains( value ) == false )
                        list.add( (float) value );
                }
            }
        }
        else if( roi instanceof FreehandRoi )
        {
            // read values from ROI using a profile plot
            // save interpolation option
            boolean interpolateOption = PlotWindow.interpolate;
            // do not interpolate pixel values
            PlotWindow.interpolate = false;
            // get label values from line roi (different from 0)
            float[] values = ( new ProfilePlot( labelImage ) )
                    .getPlot().getYValues();
            PlotWindow.interpolate = interpolateOption;

            for( int i=0; i<values.length; i++ )
            {
                if( Float.compare( 0f, values[ i ] ) != 0 &&
                        list.contains( values[ i ]) == false )
                    list.add( values[ i ]);
            }
        }
        return list;
    }





    @Override
    public void run(String s) {

        inputImage =IJ.openImage();
        supImage = IJ.openImage();
        if(inputImage == null || supImage == null){
            IJ.error("Error when opening image");
        }else {
            for(int i=0; i<numClasses; ++i){
                exampleList[i] = new java.awt.List(5);
                exampleList[i].setForeground(Color.blue);
                tags.add(new int[0]);
            }
            ArrayList<RegionFeatures.Feature> selectedFeatures = new ArrayList<>();
            selectedFeatures.add(RegionFeatures.Feature.fromLabel("Mean"));
            selectedFeatures.add(RegionFeatures.Feature.fromLabel("Median"));
            selectedFeatures.add(RegionFeatures.Feature.fromLabel("Mode"));
            selectedFeatures.add(RegionFeatures.Feature.fromLabel("Skewness"));
            selectedFeatures.add(RegionFeatures.Feature.fromLabel("Kurtosis"));
            selectedFeatures.add(RegionFeatures.Feature.fromLabel("StdDev"));
            selectedFeatures.add(RegionFeatures.Feature.fromLabel("Max"));
            selectedFeatures.add(RegionFeatures.Feature.fromLabel("Min"));
            final ArrayList<String> classes = new ArrayList<String>();
            classes.add( "class 0");
            classes.add( "class 1");
            // Define classifier
            IBk exampleClassifier = new IBk();
            trainableSuperpixelSegmentation = new TrainableSuperpixelSegmentation(inputImage,supImage,selectedFeatures,exampleClassifier,classes);
            System.out.println(trainableSuperpixelSegmentation.getFeaturesByRegion());
            win = new CustomWindow(inputImage);
            Toolbar.getInstance().setTool( Toolbar.POINT );
        }

    }
    public static void main(String[] args){
        Class<?> clazz = Trainable_Superpixel_Segmentation.class;
        String url = clazz.getResource("/" + clazz.getName().replace('.', '/') + ".class").toString();
        String pluginsDir = url.substring("file:".length(), url.length() - clazz.getName().length() - ".class".length());
        System.setProperty("plugins.dir", pluginsDir);
        IJ.runPlugIn(clazz.getName(),"");

    }


}