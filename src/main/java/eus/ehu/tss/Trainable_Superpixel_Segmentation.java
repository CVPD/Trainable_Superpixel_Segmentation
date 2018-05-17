package eus.ehu.tss;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.gui.*;
import ij.io.OpenDialog;
import ij.io.SaveDialog;
import ij.plugin.PlugIn;
import ij.process.*;
import weka.classifiers.*;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.evaluation.EvaluationUtils;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.RandomForest;
import weka.core.*;
import weka.gui.GenericObjectEditor;
import weka.gui.PropertyPanel;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.ThresholdVisualizePanel;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
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
    private ArrayList<RegionFeatures.Feature> features;
    private AbstractClassifier classifier;
    private ArrayList<String> classes;
    private TrainableSuperpixelSegmentation trainableSuperpixelSegmentation;
    private int overlay = 1;
    private Color[] colors = new Color[]{Color.red, Color.green, Color.blue, Color.cyan, Color.magenta};
    private LUT overlayLUT = null;
    private boolean calculateFeatures = true;

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
        private JButton [] addExampleButton = new JButton[500];
        private JButton addClassButton = null;
        private JButton saveClassButton = null;
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
            trainingPanel.add(trainClassButton,trainingConstraints);
            trainingConstraints.gridy++;
            overlayButton = new JButton("Toggle overlay");
            trainingPanel.add(overlayButton,trainingConstraints);
            trainingConstraints.gridy++;
            resButton = new JButton("Create result");
            trainingPanel.add(resButton,trainingConstraints);
            trainingConstraints.gridy++;
            probButton = new JButton("Get probability");
            trainingPanel.add(probButton,trainingConstraints);
            trainingConstraints.gridy++;
            plotButton = new JButton("Plot result");
            trainingPanel.add(plotButton,trainingConstraints);
            trainingConstraints.gridy++;


            //Options panel buttons
            applyClassButton = new JButton("Apply classifier");
            optionsPanel.add(applyClassButton,optionsConstraints);
            optionsConstraints.gridy++;
            loadClassButton = new JButton("Load classifier");
            optionsPanel.add(loadClassButton,optionsConstraints);
            optionsConstraints.gridy++;
            saveClassButton = new JButton("Save classifier");
            optionsPanel.add(saveClassButton,optionsConstraints);
            optionsConstraints.gridy++;
            addClassButton = new JButton("Create new class");
            optionsPanel.add(addClassButton,optionsConstraints);
            optionsConstraints.gridy++;
            settButton = new JButton("Settings");
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
            addExampleButton[0].setSelected(true);

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

            // recalculate minimum size of scroll panel
            scrollPanel.setMinimumSize( labelsJPanel.getPreferredSize() );

            repaintAll();

        }

        public void repaintAll(){
            this.annotationsPanel.repaint();
            getCanvas().repaint();
            this.controlsPanel.repaint();
            this.all.repaint();
        }
    }

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
        calculateFeatures = true;
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


    void runStopTraining(final String command){
        IJ.log("Training classifier");
        trainableSuperpixelSegmentation.setClasses(classes);
        if(calculateFeatures) {
            IJ.log("Calculating region features");
            trainableSuperpixelSegmentation.calculateRegionFeatures();
            calculateFeatures = false;
        }
        if(!trainableSuperpixelSegmentation.trainClassifier(tags)){
            IJ.error("Error when training classifier");
        }
        classifier = trainableSuperpixelSegmentation.getClassifier();
        IJ.log("Classifier trained");
        applyClassifier();
        createResult();
        overlay = 2;
        toggleOverlay();
    }

    void applyClassifier(){
        if(calculateFeatures){
            IJ.log("Calculating region features");
            trainableSuperpixelSegmentation.calculateRegionFeatures();
            calculateFeatures = false;
        }
        if(!trainableSuperpixelSegmentation.isClassifierTrained()){
            if(!trainableSuperpixelSegmentation.trainClassifier(tags)){
                IJ.error("Error when training classifier");
            }
            classifier = trainableSuperpixelSegmentation.getClassifier();
        }
        IJ.log("Applying classifier");
        if(!trainableSuperpixelSegmentation.setClassifier(classifier)){
            IJ.error("Error when setting classifier");
            return;
        }

        resultImage = trainableSuperpixelSegmentation.applyClassifier();
        IJ.log("Classifier applied");
    }

    /**
     * Save classifier to file
     */
    void saveClassifier(){

        SaveDialog sd = new SaveDialog("Save model as...","classifier",".model");
        if(sd.getFileName()==null){
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
            IJ.error("Save Failed", "Error when saving classifier into a file");
            saveOK = false;
        }
        if (saveOK)
            IJ.log("Saved model into " + filename );


    }

    /**
     * Load classifier from file
     */
    void loadClassifier(){
        OpenDialog od = new OpenDialog( "Choose Weka classifier file", "" );
        if (od.getFileName()==null)
            return;
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
            return;
        }
        classifier = loadedClassifier;
        trainableSuperpixelSegmentation.setClasses(classes);
        calculateFeatures = true;
        IJ.log(loadedClassifier.toString());

    }

    void createResult(){

        if(resultImage==null){
            if(!trainableSuperpixelSegmentation.isClassifierTrained()) {
                runStopTraining("Run");
            }else {
                applyClassifier();
            }
        }

        ImagePlus resultImg = resultImage.duplicate();


        resultImg.setTitle( "Classified image" );

        convertTo8bitNoScaling( resultImg );
        resultImg.getProcessor().setColorModel( overlayLUT );
        resultImg.getImageStack().setColorModel( overlayLUT );
        resultImg.updateAndDraw();
        resultImg.show();
    }


    void showSettingsDialog(){

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

        gd.addSlider("Overlay opacity:",0,1,win.overlayOpacity);

        gd.addMessage("Classifier options:");
        GenericObjectEditor classifierEditor = new GenericObjectEditor();
        PropertyPanel classifierEditorPanel = new PropertyPanel(classifierEditor);
        classifierEditor.setClassType(Classifier.class);
        classifierEditor.setValue(classifier);
        gd.add(classifierEditorPanel);

        Object c = (Object) classifierEditor.getValue();
        String originalOptions = "";
        String originalClassifierName = c.getClass().getName();
        if (c instanceof OptionHandler)
        {
            originalOptions = Utils.joinOptions(((OptionHandler)c).getOptions());
        }


        gd.showDialog();

        if(gd.wasCanceled()){
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
                return;
            }
            classifier = cls;
            trainableSuperpixelSegmentation.setClassifier(classifier);
            IJ.log("Current classifier: " + c.getClass().getName() + " " + options);
        }

        final double newOpacity = (double) gd.getNextNumber();
        if( newOpacity != win.overlayOpacity )
        {
            ImageRoi roi = null;
            int slice = inputImage.getCurrentSlice();
            win.overlayOpacity = newOpacity;
            if(overlay==0){
                inputImage.setOverlay(null);
            }else if(overlay==1){
                roi = new ImageRoi(0, 0, supImage.getImageStack().getProcessor(slice));
                roi.setOpacity(newOpacity);
                inputImage.setOverlay(new Overlay(roi));
            }else {
                roi = new ImageRoi(0, 0, resultImage.getImageStack().getProcessor(slice));
                roi.setOpacity(newOpacity);
                inputImage.setOverlay(new Overlay(roi));
            }
        }


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

    void showProbability(){
        trainableSuperpixelSegmentation.setClasses(classes);
        if(calculateFeatures) {
            IJ.log("Calculating region features");
            trainableSuperpixelSegmentation.calculateRegionFeatures();
            calculateFeatures = false;
        }
        if(!trainableSuperpixelSegmentation.isClassifierTrained()){
            if(!trainableSuperpixelSegmentation.trainClassifier(tags)){
                IJ.error("Error when training classifier");
            }
        }
        ImagePlus probabilityImage = trainableSuperpixelSegmentation.getProbabilityMap();
        probabilityImage.show();
    }

    void toggleOverlay(){
        int slice = inputImage.getCurrentSlice();
        ImageRoi roi = null;
        if(overlay==0){
            overlay++;
            inputImage.setOverlay(null);
        }else if(overlay==1){
            if(resultImage!=null) {
                overlay++;
            }else {
                overlay=0;
            }
            roi = new ImageRoi(0, 0, supImage.getImageStack().getProcessor(slice));
            roi.setOpacity(win.overlayOpacity);
            inputImage.setOverlay(new Overlay(roi));
        }else {
            overlay=0;
            ImagePlus resultImg = resultImage.duplicate();
            resultImg.setTitle( "Classified image" );
            convertTo8bitNoScaling( resultImg );
            resultImg.getProcessor().setColorModel( overlayLUT );
            resultImg.getImageStack().setColorModel( overlayLUT );
            ImageProcessor processor = resultImg.getImageStack().getProcessor(slice);
            roi = new ImageRoi(0, 0, processor);
            roi.setOpacity(win.overlayOpacity);
            inputImage.setOverlay(new Overlay(roi));
        }
    }


    /**
     * Delete selected tag
     * @param e action command with information about item to be deleted
     * @param i identifier of class to add tags
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
        }catch (Exception e1){
            e1.printStackTrace();
        }
        System.out.println("Deleted");
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
        ArrayList<Integer> selectedLabel = getSelectedLabels(supImage,r);
        for(Integer label: selectedLabel){
            int[] a = tags.get(i);
            //Check if tag already exists on list
            for(int x = 0;x<a.length;++x){
                if(a[x]==label){
                    IJ.log("Tag already on list");
                    return;
                }
            }
            a = Arrays.copyOf(a, a.length+1);
            a[a.length-1] = label;
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
    public static ArrayList<Integer> getSelectedLabels(
            final ImagePlus labelImage,
            final Roi roi )
    {
        final ArrayList<Integer> list = new ArrayList<Integer>();

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
                    if( Float.compare( 0f, value ) != 0 && list.contains( value ) == false ) {
                        list.add((int) value);
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
                    if( Float.compare( 0f, value ) != 0 &&
                            list.contains( value ) == false )
                        list.add( (int) value );
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
            float[] f = ( new ProfilePlot( labelImage ) ).getPlot().getYValues();
            int[] values = new int[f.length];
            for(int i=0;i<f.length;++i){
                values[i]= (int) f[i];
            }
            PlotWindow.interpolate = interpolateOption;

            for( int i=0; i<values.length; i++ )
            {
                if(Float.compare( 0f, values[i] ) == 0){
                    IJ.log("Tags with value 0 not added");
                }
                if( Float.compare( 0f, values[ i ] ) != 0 &&
                        list.contains( values[ i ]) == false )
                    list.add( values[ i ]);
            }
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
        IJ.log("Open input image");
        inputImage =IJ.openImage();
        IJ.log("Open superpixel image");
        supImage = IJ.openImage();
        if(inputImage == null || supImage == null){
            IJ.error("Error when opening image");
        }else {
            for(int i=0; i<numClasses; ++i){
                classes.add("class "+i);
                exampleList[i] = new java.awt.List(5);
                exampleList[i].setForeground(colors[i]);
                tags.add(new int[0]);
            }
            features = new ArrayList<>();
            features.add(RegionFeatures.Feature.fromLabel("Mean"));
            features.add(RegionFeatures.Feature.fromLabel("Median"));
            features.add(RegionFeatures.Feature.fromLabel("Mode"));
            features.add(RegionFeatures.Feature.fromLabel("Skewness"));
            features.add(RegionFeatures.Feature.fromLabel("Kurtosis"));
            features.add(RegionFeatures.Feature.fromLabel("StdDev"));
            features.add(RegionFeatures.Feature.fromLabel("Max"));
            features.add(RegionFeatures.Feature.fromLabel("Min"));
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


}