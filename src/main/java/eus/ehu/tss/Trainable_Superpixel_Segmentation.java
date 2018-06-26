package eus.ehu.tss;

import com.sun.scenario.effect.impl.sw.java.JSWBlend_COLOR_BURNPeer;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.WindowManager;
import ij.gui.GenericDialog;
import ij.gui.ImageCanvas;
import ij.gui.Roi;
import ij.gui.StackWindow;
import ij.gui.ImageRoi;
import ij.gui.Overlay;
import ij.gui.PointRoi;
import ij.gui.Toolbar;
import ij.io.OpenDialog;
import ij.io.SaveDialog;
import ij.plugin.PlugIn;
import ij.process.ImageConverter;
import ij.process.ImageProcessor;
import ij.process.LUT;
import ij.process.StackConverter;

import org.w3c.dom.Attr;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.evaluation.EvaluationUtils;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.Classifier;
import weka.core.Utils;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.OptionHandler;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils;
import weka.gui.GenericObjectEditor;
import weka.gui.PropertyPanel;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.ThresholdVisualizePanel;

import javax.swing.JPanel;
import javax.swing.JFrame;
import javax.swing.JScrollPane;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.BorderFactory;
import javax.swing.JOptionPane;
import javax.swing.SwingUtilities;

import java.awt.Color;
import java.awt.Panel;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.Checkbox;
import java.awt.BorderLayout;
import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.FileNotFoundException;
import java.io.File;
import java.io.OutputStream;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.io.ObjectInputStream;
import java.io.InputStream;

import java.nio.charset.StandardCharsets;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Vector;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;



public class Trainable_Superpixel_Segmentation implements PlugIn {

    private CustomWindow win;
    private ImagePlus inputImage;
    private ImagePlus supImage;
    private ImagePlus resultImage;
    private final ExecutorService exec = Executors.newFixedThreadPool(1);
    private int numClasses = 2;
    private  java.awt.List[] exampleList = new java.awt.List[500];
    private ArrayList<int[]> tags = new ArrayList<>();
    private ArrayList<HashMap<Integer,Roi>> rois = new ArrayList<>();
    private ArrayList<RegionFeatures.Feature> features;
    private AbstractClassifier classifier;
    private ArrayList<String> classes;
    private TrainableSuperpixelSegmentation trainableSuperpixelSegmentation;
    private int overlay = 1;
    private Color[] colors = new Color[]{Color.red, Color.green, Color.blue, Color.cyan, Color.magenta};
    private LUT overlayLUT = null;
    private boolean calculateFeatures = true;
    private String inputTitle;
    private boolean classBalance = true;
    private boolean trainingDataLoaded = false;

    private class CustomWindow extends StackWindow
    {
        private Panel all = new Panel();
        private JPanel controlsPanel = new JPanel();
        private JPanel optionsPanel = new JPanel();
        private JPanel trainingPanel = new JPanel();
        private JPanel annotationsPanel = new JPanel();
        private JPanel labelsJPanel = new JPanel();
        private JScrollPane scrollPanel = null;
        private GridBagConstraints annotationsConstraints;
        private GridBagLayout boxAnnotation;
        private JButton trainClassButton = null;
        private JButton loadClassButton = null;
        private JButton applyClassButton = null;
        private JButton settButton = null;
        private JButton plotButton = null;
        private JButton probButton = null;
        private JButton resButton = null;
        private JButton overlayButton = null;
        private JCheckBox overlayCheckbox = null;
        private JButton [] addExampleButton = new JButton[500];
        private JButton addClassButton = null;
        private JButton saveClassButton = null;
        private JButton saveInstButton = null;
        private JButton loadTrainingDataButton = null;
        private double overlayOpacity = 0.33;

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
                        else if(e.getSource()==addClassButton){
                            addNewClass();
                        }
                        else if(e.getSource()==saveClassButton){
                            saveClassifier();
                        }
                        else if(e.getSource()==saveInstButton){
                            saveInstances();
                        }
                        else if(e.getSource()==loadTrainingDataButton){
                            loadTrainingData();
                        }
                        else{
                            for(int i = 0; i < numClasses; i++)
                            {
                                if(e.getSource() == exampleList[i])
                                {
                                    deleteSelected(e,i);
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

            // Create overlay LUT
            final byte[] red = new byte[ 256 ];
            final byte[] green = new byte[ 256 ];
            final byte[] blue = new byte[ 256 ];

            // assign colors to classes
            colors = new Color[ 50 ];

            // hue for assigning new color ([0.0-1.0])
            float hue = 0f;
            // saturation for assigning new color ([0.5-1.0])
            float saturation = 1f;

            // first color is red: HSB( 0, 1, 1 )

            for(int i=0; i<50; i++)
            {
                colors[ i ] = Color.getHSBColor(hue, saturation, 1);

                hue += 0.38197f; // golden angle
                if (hue > 1)
                    hue -= 1;
                saturation += 0.38197f; // golden angle
                if (saturation > 1)
                    saturation -= 1;
                saturation = 0.5f * saturation + 0.5f;
            }

            for(int i = 0 ; i < 50; i++)
            {
                //IJ.log("i = " + i + " color index = " + colorIndex);
                red[i] = (byte) colors[ i ].getRed();
                green[i] = (byte) colors[ i ].getGreen();
                blue[i] = (byte) colors[ i ].getBlue();
            }
            overlayLUT = new LUT(red, green, blue);

            final ImageCanvas canvas = (ImageCanvas) getCanvas();

            //Training panel layout and constraints
            GridBagLayout trainingLayout = new GridBagLayout();
            trainingPanel.setLayout(trainingLayout);
            trainingPanel.setBorder(BorderFactory.createTitledBorder("Training"));
            GridBagConstraints trainingConstraints = new GridBagConstraints();
            trainingConstraints.anchor = GridBagConstraints.NORTHWEST;
            trainingConstraints.fill = GridBagConstraints.VERTICAL;
            trainingConstraints.gridwidth = 1;
            trainingConstraints.gridheight = 1;
            trainingConstraints.gridx = 0;
            trainingConstraints.gridy = 0;
            trainingConstraints.weightx = 0;
            trainingConstraints.weighty = 0;
            trainingConstraints.insets = new Insets(5, 5, 6, 6);

            //Options panel layout and constraints
            GridBagLayout optionsLayout = new GridBagLayout();
            optionsPanel.setLayout(optionsLayout);
            optionsPanel.setBorder(BorderFactory.createTitledBorder("Options"));
            GridBagConstraints optionsConstraints = new GridBagConstraints();
            optionsConstraints.anchor = GridBagConstraints.NORTHWEST;
            optionsConstraints.fill = GridBagConstraints.VERTICAL;
            optionsConstraints.gridwidth = 1;
            optionsConstraints.gridheight = 1;
            optionsConstraints.gridx = 0;
            optionsConstraints.gridy = 0;
            optionsConstraints.weightx = 0;
            optionsConstraints.weighty = 0;
            optionsConstraints.insets = new Insets(5, 5, 6, 6);

            //Control panel layout and constraints
            GridBagLayout controlLayout = new GridBagLayout();
            controlsPanel.setLayout(controlLayout);
            controlsPanel.setBorder(BorderFactory.createTitledBorder("Controls"));
            GridBagConstraints controlConstraints = new GridBagConstraints();
            controlConstraints.anchor = GridBagConstraints.CENTER;
            controlConstraints.fill = GridBagConstraints.VERTICAL;
            controlConstraints.gridwidth = 1;
            controlConstraints.gridheight = 1;
            controlConstraints.gridx = 0;
            controlConstraints.gridy = 0;
            controlConstraints.weightx = 0;
            controlConstraints.weighty = 0;
            controlConstraints.insets = new Insets(5, 5, 6, 6);


            // Annotations panel
            boxAnnotation = new GridBagLayout();
            annotationsPanel.setBorder(BorderFactory.createTitledBorder("Labels"));
            annotationsPanel.setLayout(boxAnnotation);
            annotationsConstraints = new GridBagConstraints();
            annotationsConstraints.anchor = GridBagConstraints.NORTHWEST;
            annotationsConstraints.fill = GridBagConstraints.HORIZONTAL;
            annotationsConstraints.gridwidth = 1;
            annotationsConstraints.gridheight = 1;
            annotationsConstraints.gridx = 0;
            annotationsConstraints.gridy = 0;

            //Labels panel (includes annotations panel)
            GridBagLayout labelsLayout = new GridBagLayout();
            GridBagConstraints labelsConstraints = new GridBagConstraints();
            labelsJPanel.setLayout( labelsLayout );
            labelsConstraints.anchor = GridBagConstraints.NORTHWEST;
            labelsConstraints.fill = GridBagConstraints.HORIZONTAL;
            labelsConstraints.gridwidth = 1;
            labelsConstraints.gridheight = 1;
            labelsConstraints.gridx = 0;
            labelsConstraints.gridy = 0;
            labelsJPanel.add( annotationsPanel, labelsConstraints );

            //Scroll panel
            scrollPanel = new JScrollPane(labelsJPanel);
            scrollPanel.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_NEVER);
            scrollPanel.setMinimumSize(labelsJPanel.getPreferredSize());

            //All panel
            GridBagLayout layout = new GridBagLayout();
            GridBagConstraints allConstraints = new GridBagConstraints();
            all.setLayout(layout);
            allConstraints.anchor = GridBagConstraints.NORTHWEST;
            allConstraints.fill = GridBagConstraints.BOTH;
            allConstraints.gridwidth = 1;
            allConstraints.gridheight = 1;
            allConstraints.gridx = 0;
            allConstraints.gridy = 0;
            allConstraints.weightx = 0;
            allConstraints.weighty = 0;
            allConstraints.insets = new Insets(5, 5, 6, 6);

            //Training panel buttons
            trainClassButton = new JButton("Train classifier");
            trainClassButton.setToolTipText("Train a classifier based on selected regions");
            trainingPanel.add(trainClassButton,trainingConstraints);
            trainingConstraints.gridy++;
            overlayButton = new JButton("Toggle overlay");
            overlayButton.setToolTipText("Toggle between superpixel image, result image and original image");
            trainingPanel.add(overlayButton,trainingConstraints);
            trainingConstraints.gridy++;
            overlayCheckbox = new JCheckBox("Display result only");
            overlayCheckbox.setToolTipText("Disable superpixel overlay showing");
            trainingPanel.add(overlayCheckbox,trainingConstraints);
            overlayCheckbox.setSelected(false);
            overlayCheckbox.setEnabled(false);
            trainingConstraints.gridy++;
            resButton = new JButton("Create result");
            resButton.setToolTipText("Create a duplicate of the result image");
            trainingPanel.add(resButton,trainingConstraints);
            trainingConstraints.gridy++;
            probButton = new JButton("Get probability");
            probButton.setToolTipText("Create a probability image");
            trainingPanel.add(probButton,trainingConstraints);
            trainingConstraints.gridy++;
            plotButton = new JButton("Plot result");
            plotButton.setToolTipText("Plot various metrics");
            trainingPanel.add(plotButton,trainingConstraints);
            trainingConstraints.gridy++;


            //Options panel buttons
            applyClassButton = new JButton("Apply classifier");
            optionsPanel.add(applyClassButton,optionsConstraints);
            applyClassButton.setToolTipText("Load an image and apply trained classifier");
            optionsConstraints.gridy++;
            loadClassButton = new JButton("Load classifier");
            loadClassButton.setToolTipText("Load Weka classifier from a file");
            optionsPanel.add(loadClassButton,optionsConstraints);
            optionsConstraints.gridy++;
            saveClassButton = new JButton("Save classifier");
            saveClassButton.setToolTipText("Save trained classifier into a file");
            optionsPanel.add(saveClassButton,optionsConstraints);
            optionsConstraints.gridy++;
            saveInstButton = new JButton("Save instances");
            saveInstButton.setToolTipText("Save training data into a Weka file");
            optionsPanel.add(saveInstButton,optionsConstraints);
            optionsConstraints.gridy++;
            loadTrainingDataButton = new JButton("Load training data");
            loadTrainingDataButton.setToolTipText("Load training data from a Weka file");
            optionsPanel.add(loadTrainingDataButton,optionsConstraints);
            optionsConstraints.gridy++;
            addClassButton = new JButton("Create new class");
            addClassButton.setToolTipText("Add a new class");
            optionsPanel.add(addClassButton,optionsConstraints);
            optionsConstraints.gridy++;
            settButton = new JButton("Settings");
            settButton.setToolTipText("Open a settings dialog");
            optionsPanel.add(settButton,optionsConstraints);
            optionsConstraints.gridy++;

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

            controlsPanel.add(trainingPanel,controlConstraints);
            controlConstraints.gridy++;

            controlsPanel.add(optionsPanel,controlConstraints);
            controlConstraints.gridy++;

            allConstraints.weightx = 1;
            allConstraints.weighty = 1;
            allConstraints.gridheight = 1;
            all.add(controlsPanel,allConstraints);
            allConstraints.gridx++;

            all.add(canvas,allConstraints);
            allConstraints.gridx++;

            allConstraints.anchor = GridBagConstraints.NORTHEAST;
            allConstraints.weightx = 0;
            allConstraints.weighty = 0;
            allConstraints.gridheight = 1;
            all.add(scrollPanel,allConstraints);
            allConstraints.gridx++;

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
            addClassButton.addActionListener(listener);
            saveClassButton.addActionListener(listener);
            saveInstButton.addActionListener(listener);
            loadTrainingDataButton.addActionListener(listener);

            GridBagLayout wingb = new GridBagLayout();
            GridBagConstraints winc = new GridBagConstraints();
            winc.anchor = GridBagConstraints.NORTHWEST;
            winc.fill = GridBagConstraints.BOTH;
            winc.weightx = 1;
            winc.weighty = 1;
            setLayout( wingb );
            add( all, winc );
            pack();
            setMinimumSize(getPreferredSize());


        }

        /**
         * Add a new class and update interface
         */
        public void addClass(){
            int classNum = numClasses;

            exampleList[classNum] = new java.awt.List(5);
            exampleList[classNum].setForeground(colors[classNum]);

            exampleList[classNum].addActionListener(listener);
            exampleList[classNum].addItemListener(itemListener);
            addExampleButton[classNum] = new JButton("Add to " + classes.get(classNum));

            annotationsConstraints.insets = new Insets(5, 5, 6, 6);

            annotationsPanel.add(addExampleButton[classNum],annotationsConstraints);
            annotationsConstraints.gridy++;

            annotationsConstraints.insets = new Insets(0,0,0,0);

            annotationsPanel.add(exampleList[classNum],annotationsConstraints);
            annotationsConstraints.gridy++;

            // Add listener to the new button
            addExampleButton[classNum].addActionListener(listener);

            numClasses++;
            tags.add(new int[0]);
            rois.add(new HashMap<>());
            // recalculate minimum size of scroll panel
            scrollPanel.setMinimumSize( labelsJPanel.getPreferredSize() );

            repaintAll();

        }

        public void setButtonsEnabled(boolean f){
            trainClassButton.setEnabled(f);
            loadClassButton.setEnabled(f);
            applyClassButton.setEnabled(f);
            settButton.setEnabled(f);
            plotButton.setEnabled(f);
            probButton.setEnabled(f);
            resButton.setEnabled(f);
            overlayButton.setEnabled(f);
            overlayCheckbox.setEnabled(f);
            for(int i=0;i<numClasses;++i){
                addExampleButton[i].setEnabled(f);
            }
            addClassButton.setEnabled(f);
            saveClassButton.setEnabled(f);
            saveInstButton.setEnabled(f);
            loadTrainingDataButton.setEnabled(f);
        }

        public void enableOverlayCheckbox(){
            overlayCheckbox.setEnabled(true);
        }

        public boolean ovCheckbox(){
            return overlayCheckbox.isSelected();
        }

        public void repaintAll(){
            this.annotationsPanel.repaint();
            getCanvas().repaint();
            this.controlsPanel.repaint();
            this.all.repaint();
        }
    }

    /**
     * Load training data from a file.
     */
    private void loadTrainingData() {
        win.setButtonsEnabled(false);
        Instances data=null;
        OpenDialog od = new OpenDialog("Choose data file", OpenDialog.getLastDirectory(), "data.arff");
        if (od.getFileName()==null) {
            win.setButtonsEnabled(true);
            return;
        }

        IJ.log("Loading data from " + od.getDirectory() + od.getFileName() + "...");
        String pathname = od.getDirectory() + od.getFileName();
        try{
            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(
                            new FileInputStream(pathname), StandardCharsets.UTF_8));
            try{
                data = new Instances(reader);
                // setting class attribute
                data.setClassIndex(data.numAttributes() - 1);
                reader.close();
            }
            catch(IOException e){
                IJ.showMessage("IOException: wrong file format!");
            }
        }
        catch(FileNotFoundException e){IJ.showMessage("File not found!");}
        try {
            if (null != data) {
                Attribute classAttribute = data.classAttribute();
                Enumeration<Object> classValues = classAttribute.enumerateValues();
                ArrayList<String> loadedClassNames = new ArrayList<String>();
                if (classAttribute.numValues() != numClasses) {
                    IJ.error("ERROR: Loaded number of classes and current number do not match!\n\tExpected number of classes: "+classAttribute.numValues()+"\n\tCurrent number of classes: "+numClasses);
                    trainingDataLoaded = false;
                    win.setButtonsEnabled(true);
                    return;
                }
                int j = 0;
                while (classValues.hasMoreElements()) {
                    final String className = ((String) classValues.nextElement()).trim();
                    loadedClassNames.add(className);

                    IJ.log("Read class name: " + className);
                    if (!className.equals(classes.get(j))) {
                        IJ.error("ERROR: Loaded classes and current classes do not match!\n\tExpected: " + className+"\n\tFound: "+classes.get(j));
                        trainingDataLoaded = false;
                        win.setButtonsEnabled(true);
                        return;
                    }
                    j++;
                }
                //Select only loaded features
                Enumeration<Attribute> attributes = data.enumerateAttributes();
                final String[] availableFeatures = RegionFeatures.Feature.getAllLabels();
                final int numFeatures = availableFeatures.length;
                boolean[] usedFeatures = new boolean[numFeatures];
                features = new ArrayList<RegionFeatures.Feature>();
                while(attributes.hasMoreElements())
                {
                    final Attribute a = attributes.nextElement();
                    String n = a.name();
                    if(a.name().endsWith("-L")||
                            a.name().endsWith("-a")||
                            a.name().endsWith("-b")){
                        n = a.name().substring(0,a.name().length()-2);
                    }
                    for(int i = 0 ; i < numFeatures; i++)
                        features.add(RegionFeatures.Feature.fromLabel(n));
                }
                if(trainableSuperpixelSegmentation.getTrainingData()!=null){
                    data = merge(trainableSuperpixelSegmentation.getTrainingData(),data);
                    IJ.log("Data merged with previous training data");
                }
                trainableSuperpixelSegmentation.setTrainingData(data);
                trainingDataLoaded = true;
                IJ.log("Data loaded");
                win.setButtonsEnabled(true);
            }

        }catch (Exception e){
            win.setButtonsEnabled(true);
            IJ.log("Error when loading training data");
            e.printStackTrace();
        }

    }

    /**
     * Add a new class based on user input, update instances if already created
     */
    void addNewClass(){
        String inputName = JOptionPane.showInputDialog("Please input a new label name");
        if(null == inputName || 0 == inputName.length()){
            IJ.error("Invalid name");
            return;
        }
        inputName.trim();
        classes.add(inputName);
        win.addClass();
        repaintWindow();
        if(!calculateFeatures){
            Instances features = trainableSuperpixelSegmentation.getUnlabeled();
            Attribute c = features.classAttribute();
            ArrayList<String> newc = new ArrayList<>();
            for(int i=0;i<c.numValues();++i){
                newc.add(c.value(i));
            }
            newc.add(inputName);
            Attribute newClass = new Attribute("class",newc);
            features.replaceAttributeAt(newClass,c.index());
            trainableSuperpixelSegmentation.setUnlabeled(features);
        }
    }

    /**
     * Repaint whole window
     */
    private void repaintWindow()
    {
        // Repaint window
        SwingUtilities.invokeLater(
                new Runnable() {
                    public void run() {
                        win.invalidate();
                        win.validate();
                        win.repaint();
                    }
                });
    }


    /**
     *Trains classifier, calculates features if needed.
     * @param command
     */
    void runStopTraining(final String command){
        win.setButtonsEnabled(false);
        if(!trainingDataLoaded) {
            for (int i = 0; i < tags.size(); ++i) {
                if (tags.get(i).length == 0) {
                    IJ.showMessage("Add at least one region to class " + classes.get(i));
                    win.setButtonsEnabled(true);
                    return;
                }
            }
        }
        IJ.log("Training classifier");
        trainableSuperpixelSegmentation.setClasses(classes);
        if(calculateFeatures) {
            IJ.log("Calculating region features");
            trainableSuperpixelSegmentation.calculateRegionFeatures();
            calculateFeatures = false;
        }
        Instances unlabeled = trainableSuperpixelSegmentation.getUnlabeled();
        ArrayList<Attribute> attributes = new ArrayList<Attribute>();
        int numFeatures = unlabeled.numAttributes()-1;
        for(int i=0;i<numFeatures;++i){
            attributes.add(new Attribute(unlabeled.attribute(i).name(),i));
        }
        attributes.add(new Attribute("Class",classes));
        Instances trainingData = new Instances("training data",attributes,0);
        // Fill training dataset with the feature vectors of the corresponding
        // regions given by classRegions
        for(int i=0;i<tags.size();++i){ //For each class in classRegions
            for(int j=0;j<tags.get(i).length;++j){
                Instance inst = new DenseInstance(numFeatures+1);
                for(int k=0;k<numFeatures;++k){
                    inst.setValue(k,unlabeled.get(tags.get(i)[j]-1).value(k));
                }
                inst.setValue(numFeatures,i); // set class value
                trainingData.add(inst);
            }
        }
        trainingData.setClassIndex(numFeatures); // set class index
        try {
            if (trainingDataLoaded) {
                trainingData = merge(trainingData, trainableSuperpixelSegmentation.getTrainingData());
                IJ.log("Merged selected data with loaded data");
            }
        }catch (Exception e){
            e.printStackTrace();
            IJ.log("Error when merging loaded training data selected data");
        }
        trainableSuperpixelSegmentation.setTrainingData(trainingData);
        if (!trainableSuperpixelSegmentation.trainClassifier()) {
            IJ.error("Error when training classifier");
        }
        classifier = trainableSuperpixelSegmentation.getClassifier();
        IJ.log("Classifier trained");
        if(!trainingDataLoaded) {
            for (int i = 0; i < tags.size(); ++i) {
                if (tags.get(i).length == 0) {
                    IJ.showMessage("Add at least one region to class " + classes.get(i));
                    win.setButtonsEnabled(true);
                    return;
                }
            }
        }
        IJ.log("Applying classifier");
        resultImage = trainableSuperpixelSegmentation.applyClassifier();
        overlay = 2;
        toggleOverlay();
        IJ.log("Classifier applied");
        win.enableOverlayCheckbox();
        win.setButtonsEnabled(true);
    }

    /**
     * Apply classifier to loaded image and corresponding superpixel image
     */
    void applyClassifier(){
        win.setButtonsEnabled(false);
        if(!trainableSuperpixelSegmentation.isClassifierTrained()){
            IJ.error("Train a classifier");
            win.setButtonsEnabled(true);
            return;
        }
        IJ.showMessage("Select input image");
        ImagePlus input = IJ.openImage();
        IJ.showMessage("Select corresponding label image");
        ImagePlus sup = IJ.openImage();
        trainableSuperpixelSegmentation.setInputImage(input);
        trainableSuperpixelSegmentation.setLabelImage(sup);
        IJ.log("Calculating region features");
        trainableSuperpixelSegmentation.calculateRegionFeatures();
        IJ.log("Applying classifier");
        ImagePlus resultImg = trainableSuperpixelSegmentation.applyClassifier();
        convertTo8bitNoScaling(resultImg);
        resultImg.getProcessor().setColorModel(overlayLUT);
        resultImg.getImageStack().setColorModel(overlayLUT);
        resultImg.updateAndDraw();
        resultImg.show();
        IJ.log("Classifier applied");
        win.setButtonsEnabled(true);
    }


    /**
     * Save instances into ARFF file
     */
    void saveInstances(){
        win.setButtonsEnabled(false);
        try {
            if(calculateFeatures){
                IJ.log("Calculating region features");
                trainableSuperpixelSegmentation.calculateRegionFeatures();
                calculateFeatures = false;
            }
            ArffSaver saver = new ArffSaver();
            Instances ins = null;
            ins = trainableSuperpixelSegmentation.getTrainingData();
            if(ins ==null){
                ins = trainableSuperpixelSegmentation.getUnlabeled();
            }
            saver.setInstances(ins);
            SaveDialog sd = new SaveDialog("Save instances as...","instances",".arff");
            if(sd.getFileName()==null){
                return;
            }
            String filename = sd.getDirectory() + sd.getFileName();
            saver.setFile(new File(filename));
            saver.writeBatch();
            IJ.log("File saved at "+sd.getDirectory()+sd.getFileName());
            win.setButtonsEnabled(true);
        } catch (IOException e) {
            win.setButtonsEnabled(true);
            e.printStackTrace();
        }

    }

    /**
     * Save classifier to file
     */
    void saveClassifier(){
        win.setButtonsEnabled(false);

        if(!trainableSuperpixelSegmentation.isClassifierTrained()){
            IJ.error("Train classifier");
            win.setButtonsEnabled(true);
            return;
        }

        SaveDialog sd = new SaveDialog("Save model as...","classifier",".model");
        if(sd.getFileName()==null){
            win.setButtonsEnabled(true);
            return;
        }
        String filename = sd.getDirectory() + sd.getFileName();
        File sFile = null;
        boolean saveOK = true;

        IJ.log("Saving model to file...");
        try {
            sFile = new File(filename);
            OutputStream os = new FileOutputStream(sFile);
            if (sFile.getName().endsWith(".gz"))
            {
                os = new GZIPOutputStream(os);
            }
            ObjectOutputStream objectOutputStream = new ObjectOutputStream(os);
            objectOutputStream.writeObject(trainableSuperpixelSegmentation.getClassifier());
            Instances trainHeader = new Instances(trainableSuperpixelSegmentation.getInstances(),0);
            objectOutputStream.writeObject(trainHeader);
            objectOutputStream.flush();
            objectOutputStream.close();
        }
        catch (Exception e)
        {
            win.setButtonsEnabled(true);
            IJ.error("Save Failed", "Error when saving classifier into a file");
            saveOK = false;
        }
        if (saveOK)
            win.setButtonsEnabled(true);
            IJ.log("Saved model into " + filename );


    }

    /**
     * Load classifier from file
     */
    void loadClassifier(){
        win.setButtonsEnabled(false);
        OpenDialog od = new OpenDialog( "Choose Weka classifier file", "" );
        if (od.getFileName()==null) {
            win.setButtonsEnabled(true);
            return;
        }
        IJ.log("Loading Weka classifier from " + od.getDirectory() + od.getFileName() + "...");
        String path = od.getDirectory() + od.getFileName();

        File selected = new File(path);
        AbstractClassifier loadedClassifier = null;
        try{
            InputStream is = new FileInputStream(selected);
            if(selected.getName().endsWith(".gz")){
                is = new GZIPInputStream(is);
            }
            ObjectInputStream objectInputStream = SerializationHelper.getObjectInputStream(is);
            loadedClassifier = (AbstractClassifier) objectInputStream.readObject();
            try{
                Instances loadedInstances = (Instances) objectInputStream.readObject();
                Attribute classAttribute = loadedInstances.classAttribute();
                Enumeration<Object> classValues = classAttribute.enumerateValues();
                int j=classAttribute.numValues();
                if(j!=numClasses){
                    int t = numClasses;
                    for(int i=0;i<t;++i){
                        classValues.nextElement();
                    }
                    for(int i=t;i<j;++i){
                        final String className = ((String)classValues.nextElement()).trim();
                        classes.add(className);
                        win.addClass();
                        repaintWindow();
                    }
                }
            }catch (Exception e){
            }
            objectInputStream.close();
        }catch (Exception e){
            IJ.log("Loading file failed: "+e.getMessage());
            win.setButtonsEnabled(true);
            return;
        }
        classifier = loadedClassifier;
        trainableSuperpixelSegmentation.setClasses(classes);
        calculateFeatures = true;
        IJ.log(loadedClassifier.toString());
        trainableSuperpixelSegmentation.setClassifierTrained(true);
        win.setButtonsEnabled(true);

    }

    /**
     *Create result image, trains classifier if it hasn't been trained before
     */
    void createResult(){
        win.setButtonsEnabled(false);
        if(resultImage==null){
            if(!trainableSuperpixelSegmentation.isClassifierTrained()) {
                runStopTraining("Run");
            }else {
                resultImage = trainableSuperpixelSegmentation.applyClassifier();
                createResult();
            }
        }else {

            ImagePlus resultImg = resultImage.duplicate();


            resultImg.setTitle(inputTitle + "-classified");

            convertTo8bitNoScaling(resultImg);
            resultImg.getProcessor().setColorModel(overlayLUT);
            resultImg.getImageStack().setColorModel(overlayLUT);
            resultImg.updateAndDraw();
            resultImg.show();
            win.enableOverlayCheckbox();
        }
        win.setButtonsEnabled(true);
    }


    /**
     * Shows dialog with feature, classifier and overlay opacity options
     */
    void showSettingsDialog(){
        win.setButtonsEnabled(false);
        GenericDialog gd = new GenericDialog("Superpixel Segmentation settings");
        gd.addMessage("Training features:");
        final int rows = RegionFeatures.totalFeatures()/2;
        final String[] avFeatures = RegionFeatures.Feature.getAllLabels();
        boolean[] enabledFeatures = new boolean[RegionFeatures.totalFeatures()];
        for(int i=0;i<RegionFeatures.totalFeatures();++i){
            if(features.contains(RegionFeatures.Feature.fromLabel(avFeatures[i]))){
                enabledFeatures[i]=true;
            }else {
                enabledFeatures[i]=false;
            }
        }
        gd.addCheckboxGroup(rows,2,avFeatures,enabledFeatures);

        if(trainingDataLoaded) {
            IJ.log("Feature selection disabled because training data has been loaded");
            Vector<Checkbox> v = gd.getCheckboxes();
            for (int i = 0; i < v.size(); i++) {
                v.get(i).setEnabled(false);
            }
        }

        gd.addSlider("Overlay opacity:",0,1,win.overlayOpacity);

        gd.addCheckbox("Balance classes",classBalance);

        // classifier options
        gd.addMessage( "Classifier options:" );
        GenericObjectEditor classifierEditor = new GenericObjectEditor();
        PropertyPanel classifierEditorPanel = new PropertyPanel(classifierEditor);
        classifierEditor.setClassType(Classifier.class);
        classifierEditor.setValue(classifier);
        Panel helperPanel = new Panel();
        helperPanel.add( classifierEditorPanel );
        gd.addPanel( helperPanel );

        Object c = (Object) classifierEditor.getValue();
        String originalOptions = "";
        String originalClassifierName = c.getClass().getName();
        if (c instanceof OptionHandler)
        	originalOptions = Utils.joinOptions(((OptionHandler)c).getOptions());


        gd.addMessage("Class names:");
        for(int i = 0; i < numClasses; i++)
            gd.addStringField("Class "+(i+1), classes.get(i), 15);

        gd.showDialog();

        if(gd.wasCanceled()){
            win.setButtonsEnabled(true);
            return;
        }
        ArrayList<RegionFeatures.Feature> newFeatures = new ArrayList<RegionFeatures.Feature>();
        for(int i=0;i<RegionFeatures.totalFeatures();++i){
            enabledFeatures[i] = gd.getNextBoolean();
            if(enabledFeatures[i]){
                newFeatures.add(RegionFeatures.Feature.fromLabel(avFeatures[i]));
            }
        }
        if(!features.equals(newFeatures)){
            calculateFeatures=true;
        }
        features = newFeatures;
        trainableSuperpixelSegmentation.setSelectedFeatures(features);

        boolean newBalance = gd.getNextBoolean();

        if(newBalance!=classBalance) {
            classBalance = newBalance;
            trainableSuperpixelSegmentation.setBalanceClasses(classBalance);
        }

        // check classifier options
        c = (Object)classifierEditor.getValue();
        String options = "";
        final String[] optionsArray = ((OptionHandler)c).getOptions();
        if (c instanceof OptionHandler)
        {
            options = Utils.joinOptions( optionsArray );
        }
        if( !originalClassifierName.equals( c.getClass().getName() )
                || !originalOptions.equals( options ) )
        {
            AbstractClassifier cls;
            try{
                cls = (AbstractClassifier) (c.getClass().newInstance());
                cls.setOptions( optionsArray );
            }
            catch(Exception ex)
            {
                ex.printStackTrace();
                win.setButtonsEnabled(true);
                return;
            }
            classifier = cls;
            trainableSuperpixelSegmentation.setClassifier(classifier);
            IJ.log("Current classifier: " + c.getClass().getName() + " " + options);
        }

        final double newOpacity = (double) gd.getNextNumber();
        if( newOpacity != win.overlayOpacity )
        {
            win.overlayOpacity = newOpacity;
            int slice = inputImage.getCurrentSlice();
            ImageRoi roi = null;
            if(inputImage.getOverlay()!=null){
               if(overlay==0&&resultImage!=null){
                   ImagePlus resultImg = resultImage.duplicate();
                   convertTo8bitNoScaling( resultImg );
                   resultImg.getProcessor().setColorModel( overlayLUT );
                   resultImg.getImageStack().setColorModel( overlayLUT );
                   ImageProcessor processor = resultImg.getImageStack().getProcessor(slice);
                   roi = new ImageRoi(0, 0, processor);
                   roi.setOpacity(win.overlayOpacity);
                   inputImage.setOverlay(new Overlay(roi));
               }else{
                   roi = new ImageRoi(0, 0, supImage.getImageStack().getProcessor(slice));
                   roi.setOpacity(win.overlayOpacity);
                   inputImage.setOverlay(new Overlay(roi));
               }
            }else{
                roi = new ImageRoi(0, 0, supImage.getImageStack().getProcessor(slice));
                roi.setOpacity(win.overlayOpacity);
                inputImage.setOverlay(new Overlay(roi));
                if(resultImage!=null) {
                    overlay++;
                }else {
                    overlay=0;
                }
            }
        }
        boolean classNameChanged = false;
        for(int i = 0; i < numClasses; i++)
        {
            String s = gd.getNextString();
            if (null == s || 0 == s.length()) {
                IJ.log("Invalid name for class " + (i+1));
                continue;
            }
            s = s.trim();
            if(!s.equals(classes.get(i)))
            {
                if (0 == s.toLowerCase().indexOf("add to ")) {
                    s = s.substring(7);
                }

                classes.set(i,s);
                classNameChanged = true;
                win.addExampleButton[i].setText("Add to " + s);
            }
        }
        win.setButtonsEnabled(true);


    }

    /**
     * Taken from Weka_Segmentation
     * Plot the current result
     */
    void showPlot(){

        IJ.showStatus("Evaluating current data...");
        IJ.log("Evaluating current data...");
        final Instances data;
        if(trainableSuperpixelSegmentation.getTrainingData()!=null){
            data = trainableSuperpixelSegmentation.getTrainingData();
        }else {
            data = null;
        }
        if(null == data)
        {
            IJ.error( "Train classifier");
            return;
        }
        displayGraphs(data,trainableSuperpixelSegmentation.getClassifier());
        IJ.showStatus("Done.");
        IJ.log("Done");
    }

    /**
     * Taken from Weka_Segmentation
     * Display the threshold curve window (for precision/recall, ROC, etc.).
     *
     * @param data input instances
     * @param classifier classifier to evaluate
     */
    public static void displayGraphs(Instances data, AbstractClassifier classifier)
    {
        ThresholdCurve tc = new ThresholdCurve();

        ArrayList<Prediction> predictions = null;
        try {
            final EvaluationUtils eu = new EvaluationUtils();
            predictions = eu.getTestPredictions(classifier, data);
        } catch (Exception e) {
            IJ.log("Error while evaluating data!");
            e.printStackTrace();
            return;
        }

        Instances result = tc.getCurve(predictions);
        ThresholdVisualizePanel vmc = new ThresholdVisualizePanel();
        vmc.setName(result.relationName() + " (display only)");
        PlotData2D tempd = new PlotData2D(result);
        tempd.setPlotName(result.relationName());
        tempd.addInstanceNumberAttribute();
        try {
            vmc.addPlot(tempd);
        } catch (Exception e) {
            IJ.log("Error while adding plot to visualization panel!");
            e.printStackTrace();
            return;
        }
        String plotName = vmc.getName();
        JFrame jf = new JFrame("Weka Classifier Visualize: "+plotName);
        jf.setSize(500,400);
        jf.getContentPane().setLayout(new BorderLayout());
        jf.getContentPane().add(vmc, BorderLayout.CENTER);
        jf.setVisible(true);
    }

    /**
     * Creates and displays probability maps, calculates features if they haven't been calculated
     */
    void showProbability(){
        win.setButtonsEnabled(false);
        if(!trainingDataLoaded) {
            for (int i = 0; i < tags.size(); ++i) {
                if (tags.get(i).length == 0) {
                    IJ.showMessage("Add at least one region to class " + classes.get(i));
                    return;
                }
            }
        }
        trainableSuperpixelSegmentation.setClasses(classes);
        if(calculateFeatures) {
            IJ.log("Calculating region features");
            trainableSuperpixelSegmentation.calculateRegionFeatures();
            calculateFeatures = false;
        }
        if(!trainableSuperpixelSegmentation.isClassifierTrained()){
            Instances unlabeled = trainableSuperpixelSegmentation.getUnlabeled();
            ArrayList<Attribute> attributes = new ArrayList<Attribute>();
            int numFeatures = unlabeled.numAttributes()-1;
            for(int i=0;i<numFeatures;++i){
                attributes.add(new Attribute(unlabeled.attribute(i).name(),i));
            }
            attributes.add(new Attribute("Class",classes));
            Instances trainingData = new Instances("training data",attributes,0);
            // Fill training dataset with the feature vectors of the corresponding
            // regions given by classRegions
            for(int i=0;i<tags.size();++i){ //For each class in classRegions
                for(int j=0;j<tags.get(i).length;++j){
                    Instance inst = new DenseInstance(numFeatures+1);
                    for(int k=0;k<numFeatures;++k){
                        inst.setValue(k,unlabeled.get(tags.get(i)[j]-1).value(k));
                    }
                    inst.setValue(numFeatures,i); // set class value
                    trainingData.add(inst);
                }
            }
            trainingData.setClassIndex(numFeatures); // set class index
            try {
                if (trainingDataLoaded) {
                    trainingData = merge(trainingData, trainableSuperpixelSegmentation.getTrainingData());
                    IJ.log("Merged selected data with loaded data");
                }
            }catch (Exception e){
                e.printStackTrace();
                IJ.log("Error when merging loaded training data selected data");
            }
            trainableSuperpixelSegmentation.setTrainingData(trainingData);
            if (!trainableSuperpixelSegmentation.trainClassifier()) {
                IJ.error("Error when training classifier");
            }
        }
        ImagePlus probabilityImage = trainableSuperpixelSegmentation.getProbabilityMap();
        probabilityImage.setTitle(inputTitle+"-prob");
        probabilityImage.show();
        win.setButtonsEnabled(true);

    }

    /**
     * Changes displayed overlay, between no overlay, superpixel overlay and result overlay
     */
    void toggleOverlay(){
        int slice = inputImage.getCurrentSlice();
        ImageRoi roi = null;
        if(overlay==0){
            overlay++;
            inputImage.setOverlay(null);
        }else {
            if(win.ovCheckbox()) {
                overlay = 0;
                ImagePlus resultImg = resultImage.duplicate();
                convertTo8bitNoScaling(resultImg);
                resultImg.getProcessor().setColorModel(overlayLUT);
                resultImg.getImageStack().setColorModel(overlayLUT);
                ImageProcessor processor = resultImg.getImageStack().getProcessor(slice);
                roi = new ImageRoi(0, 0, processor);
                roi.setOpacity(win.overlayOpacity);
                inputImage.setOverlay(new Overlay(roi));
            }else {
                if (overlay == 1) {
                    if (resultImage != null) {
                        overlay++;
                    } else {
                        overlay = 0;
                    }
                    roi = new ImageRoi(0, 0, supImage.getImageStack().getProcessor(slice));
                    roi.setOpacity(win.overlayOpacity);
                    inputImage.setOverlay(new Overlay(roi));
                } else {
                    overlay = 0;
                    ImagePlus resultImg = resultImage.duplicate();
                    convertTo8bitNoScaling(resultImg);
                    resultImg.getProcessor().setColorModel(overlayLUT);
                    resultImg.getImageStack().setColorModel(overlayLUT);
                    ImageProcessor processor = resultImg.getImageStack().getProcessor(slice);
                    roi = new ImageRoi(0, 0, processor);
                    roi.setOpacity(win.overlayOpacity);
                    inputImage.setOverlay(new Overlay(roi));
                }
            }
        }
    }


    /**
     * Delete selected tag
     * @param e action command with information about item to be deleted
     * @param i identifier of class to remove tags
     */
    void deleteSelected(final ActionEvent e, final int i){
        int f = Integer.parseInt(e.getActionCommand());
        int item = f;
        int[] a = tags.get(i);
        int[] b = new int[a.length-1];
        boolean post=false;
        for(int x=0;x<a.length;++x){
            if(a[x]!=item){
                if(post){
                    b[x-1]=a[x];
                }else {
                    b[x]=a[x];
                }
            }else {
                post=true;
            }
        }
        tags.set(i,b);
        try {
            exampleList[i].remove(Integer.toString(f));
            rois.get(i).remove(f);
        }catch (Exception e1){
            e1.printStackTrace();
        }
    }

    /**
     * Adds tags based on the ROIs selected by the user
     * @param i identifier of class to add tags
     */
    void addExamples(int i){

        final Roi r = inputImage.getRoi();
        if(null == r){
            IJ.log("Select a ROI before adding");
            return;
        }
        inputImage.killRoi();
        HashMap<Integer,Roi> selectedLabel = getSelectedLabels(supImage,r);
        int y=0;
        for(int key : selectedLabel.keySet()){
            boolean dup = false;
            int[] a = tags.get(i);
            //Check if tag already exists on list
            for(int x = 0;x<a.length;++x){
                if(a[x]==key){
                    IJ.log("Tag already on list");
                    dup = true;
                    break;
                }
            }
            if(!dup) {
                a = Arrays.copyOf(a, a.length + 1);
                a[a.length - 1] = key;
                tags.set(i, a);
                exampleList[i].add(Integer.toString(key));
                rois.get(i).put(key,selectedLabel.get(key));
            }
        }
    }

    /**
     * Displays selected Roi
     * @param e
     * @param i
     */
    void listSelected(final ItemEvent e, final int i)
    {
        for(int j=0;j<numClasses;++j){
            if(j==i){
                int selectedIndex  = exampleList[i].getSelectedIndex();
                int key = Integer.parseInt(exampleList[i].getItem(selectedIndex));
                HashMap<Integer,Roi> a = rois.get(i);
                final Roi newroi = a.get(key);
                newroi.setImage(inputImage);
                inputImage.setRoi(newroi);
            }else{
                exampleList[j].deselect(exampleList[j].getSelectedIndex());
            }
        }
        inputImage.updateAndDraw();
    }

    /**
     * Modified method taken from MorphoLibJ, LabelImages class
     *
     * Get list of selected labels in label image. Labels are selected by
     * either a freehand ROI or point ROIs. Zero-value label is skipped.
     *
     * @param labelImage  label image
     * @param roi  FreehandRoi or PointRoi with selected labels
     * @return list of selected labels
     */
    public static HashMap<Integer,Roi> getSelectedLabels(
            final ImagePlus labelImage,
            final Roi roi )
    {
        final HashMap<Integer,Roi> list = new HashMap<>();

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
                    if(Float.compare( 0f, value ) == 0){
                        IJ.log("Tag with value 0 not added");
                    }
                    if( Float.compare( 0f, value ) != 0) {
                        list.putIfAbsent((int) value, new Roi(xpoints[i],ypoints[i],0,0));
                    }
                }
            }
            else
            {
                final ImageProcessor ip = labelImage.getProcessor();
                for ( int i = 0; i<xpoints.length; i ++ )
                {
                    float value = ip.getf( xpoints[ i ], ypoints[ i ]);
                    if(Float.compare( 0f, value ) == 0){
                        IJ.log("Tag with value 0 not added");
                    }
                    if( Float.compare( 0f, value ) != 0)
                        list.putIfAbsent((int) value, new Roi(xpoints[i],ypoints[i],0,0));
                }
            }
        }else {
            IJ.error("Please use multipoint selection");
            Toolbar.getInstance().setTool( Toolbar.POINT );
        }
        return list;
    }


    /**
     * Taken from Weka_Segmentation
     *
     * Convert image to 8 bit in place without scaling it
     *
     * @param image input image
     */
    static void convertTo8bitNoScaling( ImagePlus image )
    {
        boolean aux = ImageConverter.getDoScaling();

        ImageConverter.setDoScaling( false );

        if( image.getImageStackSize() > 1)
            (new StackConverter( image )).convertToGray8();
        else
            (new ImageConverter( image )).convertToGray8();

        ImageConverter.setDoScaling( aux );
    }


    @Override
    public void run(String s) {

    	classes = new ArrayList<String>();
    	// Check number of open images
    	int nbima = WindowManager.getImageCount();
    	// If less than 2 images opened, ask user
    	if( nbima < 2 )
    	{
    		IJ.log("Open input image");
    		inputImage = IJ.openImage();
    		inputTitle = inputImage.getTitle();
    		IJ.log("Open superpixel image");
    		supImage = IJ.openImage();
    	}
    	else // otherwise read currently opened image titles
    	{
    		String[] names = new String[ nbima ];

    		for (int i = 0; i < nbima; i++)
    			names[ i ] = WindowManager.getImage(i + 1).getShortTitle();

    		GenericDialog gd = new GenericDialog( "Trainable Superpixel Segmentation" );
    		gd.addChoice( "Input image", names, names[ 0 ] );
    		gd.addChoice( "Superpixel image", names, names[ 1 ] );

    		gd.showDialog();

    		if( gd.wasOKed() )
    		{
    			int inputIndex = gd.getNextChoiceIndex();
    			int supIndex = gd.getNextChoiceIndex();
    			inputImage = WindowManager.getImage( inputIndex + 1 ).duplicate();
    			supImage = WindowManager.getImage( supIndex + 1 ).duplicate();

    			if( inputImage.getWidth() != supImage.getWidth() ||
    					inputImage.getHeight() != supImage.getHeight() ||
    					inputImage.getImageStackSize() != supImage.getImageStackSize() )
    			{
    				IJ.error( "Trainable Superpixel Segmentation input error", "Error: input"
    						+ " and superpixel images must have the same size" );
    				return;
    			}
    		}
    		else
    			return;
    	}

    	inputImage.setTitle("Trainable Superpixel Segmentation");

        if(inputImage == null || supImage == null){
            IJ.error("Error when opening image");
        }else {
            for(int i=0; i<numClasses; ++i){
                classes.add("class "+i);
                exampleList[i] = new java.awt.List(5);
                exampleList[i].setForeground(colors[i]);
                tags.add(new int[0]);
                rois.add(new HashMap<>());
            }
            features = new ArrayList<>();
            String[] selectedFs = RegionFeatures.Feature.getAllLabels();
            for(int i=0;i<selectedFs.length;++i){
                features.add(RegionFeatures.Feature.fromLabel(selectedFs[i]));
            }
            // Define classifier
            classifier = new RandomForest();
            trainableSuperpixelSegmentation = new TrainableSuperpixelSegmentation(inputImage,supImage,features,classifier,classes);
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

    public static Instances merge(Instances data1, Instances data2) throws Exception {
        int asize = data1.numAttributes();
        boolean[] strings_pos = new boolean[asize];

        for(int i = 0; i < asize; ++i) {
            Attribute att = data1.attribute(i);
            strings_pos[i] = att.type() == 2 || att.type() == 1;
        }

        Instances dest = new Instances(data1);
        dest.setRelationName(data1.relationName() + "+" + data2.relationName());
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(data2);
        Instances instances = source.getStructure();
        Instance instance = null;

        while(source.hasMoreElements(instances)) {
            instance = source.nextElement(instances);
            dest.add(instance);

            for(int i = 0; i < asize; ++i) {
                if(strings_pos[i]) {
                    dest.instance(dest.numInstances() - 1).setValue(i, instance.stringValue(i));
                }
            }
        }

        return dest;
    }


}