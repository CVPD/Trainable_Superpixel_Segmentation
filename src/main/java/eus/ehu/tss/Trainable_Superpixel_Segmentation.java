package eus.ehu.tss;

import ij.IJ;
import ij.ImagePlus;
import ij.WindowManager;
import ij.gui.GenericDialog;
import ij.gui.ImageCanvas;
import ij.gui.Roi;
import ij.gui.StackWindow;
import ij.gui.ImageRoi;
import ij.gui.Overlay;
import ij.gui.Toolbar;
import ij.io.OpenDialog;
import ij.io.SaveDialog;
import ij.measure.ResultsTable;
import ij.plugin.PlugIn;
import ij.process.ImageProcessor;
import ij.process.LUT;
import ij.process.ColorProcessor;


import inra.ijpb.label.LabelImages;
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

import java.awt.event.AdjustmentListener;
import java.awt.event.AdjustmentEvent;
import java.awt.event.MouseWheelEvent;
import java.awt.event.MouseWheelListener;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.BorderLayout;
import java.awt.Checkbox;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.Panel;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
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

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Vector;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;



/**
 * Main class of the Trainable Superpixel Segmentation plugin.
 */
public class Trainable_Superpixel_Segmentation implements PlugIn {

    private CustomWindow win;
    /** input image (2D or stack) */
    private ImagePlus inputImage;
    /** superpixel image associated to input image */
    private ImagePlus supImage;
    /** result image */
    private ImagePlus resultImage;
    private final ExecutorService exec = Executors.newFixedThreadPool(1);
    private int numClasses = 2;
    private java.awt.List[] displayedLists = new java.awt.List[500];
    /** list of lists of annotations (ROIs) */
    private ArrayList<ArrayList<Roi>>[] aRoiList;
    private ArrayList<RegionFeatures.Feature> features;
    private AbstractClassifier classifier;
    private ArrayList<String> classes;
    private TrainableSuperpixelSegmentation trainableSuperpixelSegmentation;
    private Color[] colors = new Color[]{Color.red, Color.green, Color.blue, Color.cyan, Color.magenta};
    private LUT overlayLUT = null;
    private boolean calculateFeatures = true;
    private String inputTitle;
    private boolean classBalance = true;
    private boolean trainingDataLoaded = false;
    private Instances loadedTrainingData = null;
    private Thread trainingTask = null;

    /** enumeration of overlay modes */
	public static enum OverlayMode {
		/**	display mode with transparent color traces on top of original image */
		INPUT_IMAGE,
		/** display mode with segmentation result on top of original image */
		RESULT_IMAGE,
		/** display mode with transparent superpixel image on top of original image */
		SUPERPIXEL_IMAGE };
	OverlayMode currentOverlay = OverlayMode.INPUT_IMAGE;
    /**
     * Custom window to define GUI.
     *
     */
    private class CustomWindow extends StackWindow
    {
        /** Serial version UID */
		private static final long serialVersionUID = -3132204621292464332L;
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
        private int state = 0;

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
                                if(e.getSource() == displayedLists[i])
                                {
                                    deleteSelected( i );
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
                            if(e.getSource() == displayedLists[i])
                                listSelected(e, i);
                        }
                    }
                });
            }
        };
        /**
         * Mouse listener to allow deleting selected traces by right-clicking.
         */
        private MouseListener mouseListener = new MouseListener() {
			@Override
			public void mouseReleased(MouseEvent e) {}
			@Override
			public void mousePressed(MouseEvent e) {}
			@Override
			public void mouseExited(MouseEvent e) {}
			@Override
			public void mouseEntered(MouseEvent e) {}
			@Override
			public void mouseClicked(MouseEvent me) {
				// Right click
				if(me.getButton() == MouseEvent.BUTTON3) {
					java.awt.List list = (java.awt.List) me.getSource();
					if( list.getSelectedIndex() != -1 )
					{
						int answer = JOptionPane.showConfirmDialog(
								list, "Delete selected trace?", "Delete trace", JOptionPane.YES_NO_OPTION );
						if( answer == JOptionPane.YES_OPTION )
							for(int i = 0; i < numClasses; i++)
							{
								if( list == displayedLists[i])
								{
									deleteSelected( i );
									break;
								}
							}
					}
				}
			}
		};
        /**
         * GUI constructor.
         */
        CustomWindow(ImagePlus imp)
        {
            super(imp, new ImageCanvas(imp));

            // Check image dimensions to avoid a GUI larger than the
            // screen dimensions
            Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
            double screenWidth = screenSize.getWidth();
            double screenHeight = screenSize.getHeight();

            // Zoom in if image is too small
            while( ( ic.getWidth() < screenWidth/2 ||
            		ic.getHeight() < screenHeight/2 ) &&
            		ic.getMagnification() < 32.0 )
            {
            	final int canvasWidth = ic.getWidth();
            	ic.zoomIn( 0, 0 );
            	// check if canvas size changed (otherwise stop zooming)
            	if( canvasWidth == ic.getWidth() )
            	{
            		ic.zoomOut(0, 0);
            		break;
            	}
            }
            // Zoom out if canvas is too large
            while( ( ic.getWidth() > 0.75 * screenWidth ||
            		ic.getHeight() > 0.75 * screenHeight ) &&
            		ic.getMagnification() > 1/72.0 )
            {
            	final int canvasWidth = ic.getWidth();
            	ic.zoomOut( 0, 0 );
            	// check if canvas size changed (otherwise stop zooming)
            	if( canvasWidth == ic.getWidth() )
            	{
            		ic.zoomIn(0, 0);
            		break;
            	}
            }

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
            controlConstraints.anchor = GridBagConstraints.NORTHWEST;
            controlConstraints.fill = GridBagConstraints.HORIZONTAL;
            controlConstraints.gridwidth = 1;
            controlConstraints.gridheight = 1;
            controlConstraints.gridx = 0;
            controlConstraints.gridy = 0;
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

            // Fill annotation panel
            for (int i = 0; i < numClasses; ++i) {
            	displayedLists[i].addMouseListener(mouseListener);
            	displayedLists[i].addActionListener(listener);
            	displayedLists[i].addItemListener(itemListener);
            	addExampleButton[i] = new JButton("Add to class " + i);
            	addExampleButton[i].setToolTipText("Add markings of label 'class " + i + "'");

            	annotationsConstraints.insets = new Insets(5, 5, 6, 6);

            	annotationsPanel.add(addExampleButton[i], annotationsConstraints);
            	annotationsConstraints.gridy++;

            	annotationsConstraints.insets = new Insets(0, 0, 0, 0);

            	annotationsPanel.add(displayedLists[i], annotationsConstraints);
            	annotationsConstraints.gridy++;
            }

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
            saveInstButton = new JButton("Save training data");
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

            controlsPanel.add(trainingPanel,controlConstraints);
            controlConstraints.gridy++;

            controlsPanel.add(optionsPanel,controlConstraints);
            controlConstraints.gridy++;

            all.add(controlsPanel,allConstraints);

            // put canvas in place
            allConstraints.gridx++;
            allConstraints.weightx = 1;
            allConstraints.weighty = 1;
            all.add(canvas,allConstraints);

            allConstraints.gridy++;
            allConstraints.weightx = 0;
            allConstraints.weighty = 0;
            if( null != super.sliceSelector )
            {
                sliceSelector.setValue( inputImage.getCurrentSlice() );
                supImage.setSlice( inputImage.getCurrentSlice() );

                all.add( super.sliceSelector, allConstraints );
                if( null != super.zSelector ) {
                    all.add(super.zSelector, allConstraints);
                }
                if( null != super.tSelector ){
                    all.add( super.tSelector, allConstraints );
                }
                if( null != super.cSelector ){
                    all.add( super.cSelector, allConstraints );
                }
            }
            allConstraints.gridy--;


            allConstraints.gridx++;
            allConstraints.anchor = GridBagConstraints.NORTHEAST;
            allConstraints.weightx = 0;
            allConstraints.weighty = 0;
            allConstraints.gridheight = 1;
            all.add(scrollPanel,allConstraints);

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

            if(null != sliceSelector) {
                // set slice selector to the correct number
                sliceSelector.setValue(imp.getSlice());
                // add adjustment listener to the scroll bar
                sliceSelector.addAdjustmentListener(new AdjustmentListener() {

                    public void adjustmentValueChanged(final AdjustmentEvent e) {
                        exec.submit(new Runnable() {
                            public void run() {
                                if (e.getSource() == sliceSelector) {
                                    inputImage.killRoi();
                                    updateOverlay();
                                    updateDisplayedLists();
                                }

                            }
                        });

                    }
                });
                addMouseWheelListener(new MouseWheelListener() {

                    @Override
                    public void mouseWheelMoved(final MouseWheelEvent e) {

                        exec.submit(new Runnable() {
                            public void run()
                            {
                                //IJ.log("moving scroll");
                                inputImage.killRoi();
                                updateOverlay();
                                updateDisplayedLists();
                            }
                        });

                    }
                });

                // key listener to repaint the display image and the traces
                // when using the keys to scroll the stack
                KeyListener keyListener = new KeyListener() {

                    @Override
                    public void keyTyped(KeyEvent e) {}

                    @Override
                    public void keyReleased(final KeyEvent e) {
                        exec.submit(new Runnable() {
                            public void run()
                            {
                                if(e.getKeyCode() == KeyEvent.VK_LEFT ||
                                        e.getKeyCode() == KeyEvent.VK_RIGHT ||
                                        e.getKeyCode() == KeyEvent.VK_LESS ||
                                        e.getKeyCode() == KeyEvent.VK_GREATER ||
                                        e.getKeyCode() == KeyEvent.VK_COMMA ||
                                        e.getKeyCode() == KeyEvent.VK_PERIOD)
                                {
                                    //IJ.log("moving scroll");
                                    inputImage.killRoi();
                                    updateOverlay();
                                    updateDisplayedLists();
                                }
                            }
                        });

                    }

                    @Override
                    public void keyPressed(KeyEvent e) {}
                };
                // add key listener to the window and the canvas
                addKeyListener(keyListener);
                canvas.addKeyListener(keyListener);
            }

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
    	 * Update the current overlay on the displayed image.
    	 */
        public void updateOverlay(){
        	int slice = inputImage.getCurrentSlice();
            ImageRoi roi = null;
            inputImage.setOverlay(null);
        	switch( currentOverlay )
        	{
        		case INPUT_IMAGE:
        			win.updateDisplayedLists();
        			break;
        		case RESULT_IMAGE:
        			ImagePlus resultImg = resultImage.duplicate();
                    eus.ehu.tss.Utils.convertTo8bitNoScaling(resultImg);
                    resultImg.getProcessor().setColorModel(overlayLUT);
                    resultImg.getImageStack().setColorModel(overlayLUT);
                    ImageProcessor processor = resultImg.getImageStack().getProcessor(slice);
                    roi = new ImageRoi(0, 0, processor);
                    roi.setOpacity(win.overlayOpacity);
                    inputImage.setOverlay(new Overlay(roi));
                    break;
        		case SUPERPIXEL_IMAGE:
        			roi = new ImageRoi(0, 0, supImage.getImageStack().getProcessor(slice));
                    roi.setOpacity(win.overlayOpacity);
                    inputImage.setOverlay(new Overlay(roi));
                    break;
        		default:
        			break;
        	}
        	// Overlay traces of the current slice
        	win.overlayTraces();
        }

        /**
         * Add a new class and update interface
         */
        public void addClass(){
            int classNum = numClasses;
            displayedLists[classNum] = new java.awt.List(5);
            displayedLists[classNum].setForeground(colors[classNum]);

            displayedLists[classNum].addActionListener(listener);
            displayedLists[classNum].addItemListener(itemListener);
            addExampleButton[classNum] = new JButton("Add to " + classes.get(classNum));

            annotationsConstraints.fill = GridBagConstraints.HORIZONTAL;
            annotationsConstraints.insets = new Insets(5, 5, 6, 6);

            boxAnnotation.setConstraints(addExampleButton[classNum], annotationsConstraints);
            annotationsPanel.add(addExampleButton[classNum], annotationsConstraints);
            annotationsConstraints.gridy++;

            annotationsConstraints.insets = new Insets(0, 0, 0, 0);

            boxAnnotation.setConstraints(displayedLists[classNum], annotationsConstraints);
            annotationsPanel.add(displayedLists[classNum], annotationsConstraints);
            annotationsConstraints.gridy++;

            // Add listener to the new button
            addExampleButton[classNum].addActionListener(listener);
            for(int l=0;l<inputImage.getNSlices();++l) {
                aRoiList[l].add(new ArrayList<>());
            }

            numClasses++;
            // recalculate minimum size of scroll panel
            scrollPanel.setMinimumSize( labelsJPanel.getPreferredSize() );

            repaintAll();

        }

        /**
         * State 0: No classifier trained, no result created
         * State 1: Classifier loaded, no result generated
         * Other: Classifier trained and result generated
         * @param state
         */
        public void setButtonsEnabled(int state){
            if(win.state!=0&&state==0){
                state=win.state;
            }
            if(state==0){
                trainClassButton.setEnabled(true);
                loadClassButton.setEnabled(true);
                applyClassButton.setEnabled(false);
                settButton.setEnabled(true);
                plotButton.setEnabled(false);
                probButton.setEnabled(false);
                resButton.setEnabled(false);
                overlayButton.setEnabled(true);
                overlayCheckbox.setEnabled(false);
                for(int i=0;i<numClasses;++i){
                    addExampleButton[i].setEnabled(true);
                }
                addClassButton.setEnabled(true);
                saveClassButton.setEnabled(false);
                saveInstButton.setEnabled(false);
                loadTrainingDataButton.setEnabled(true);
            }else if(state==1){
                trainClassButton.setEnabled(true);
                loadClassButton.setEnabled(true);
                applyClassButton.setEnabled(true);
                settButton.setEnabled(true);
                if(trainingDataLoaded) {
                    plotButton.setEnabled(true);
                }else {
                    plotButton.setEnabled(false);
                }
                probButton.setEnabled(false);
                resButton.setEnabled(true);
                overlayButton.setEnabled(true);
                overlayCheckbox.setEnabled(false);
                for(int i=0;i<numClasses;++i){
                    addExampleButton[i].setEnabled(true);
                }
                addClassButton.setEnabled(true);
                saveClassButton.setEnabled(true);
                saveInstButton.setEnabled(false);
                loadTrainingDataButton.setEnabled(true);
                if(win.state!=2){
                    win.state=1;
                }
            }else{
                trainClassButton.setEnabled(true);
                loadClassButton.setEnabled(true);
                applyClassButton.setEnabled(true);
                settButton.setEnabled(true);
                plotButton.setEnabled(true);
                probButton.setEnabled(true);
                resButton.setEnabled(true);
                overlayButton.setEnabled(true);
                overlayCheckbox.setEnabled(true);
                for(int i=0;i<numClasses;++i){
                    addExampleButton[i].setEnabled(true);
                }
                addClassButton.setEnabled(true);
                saveClassButton.setEnabled(true);
                saveInstButton.setEnabled(true);
                loadTrainingDataButton.setEnabled(true);
                win.state=2;
            }
        }

        /**
         * Disable all buttons of GUI
         */
        public void disableAllButtons(){
            trainClassButton.setEnabled(false);
            loadClassButton.setEnabled(false);
            applyClassButton.setEnabled(false);
            settButton.setEnabled(false);
            plotButton.setEnabled(false);
            probButton.setEnabled(false);
            resButton.setEnabled(false);
            overlayButton.setEnabled(false);
            overlayCheckbox.setEnabled(false);
            for(int i=0;i<numClasses;++i){
                addExampleButton[i].setEnabled(false);
            }
            addClassButton.setEnabled(false);
            saveClassButton.setEnabled(false);
            saveInstButton.setEnabled(false);
            loadTrainingDataButton.setEnabled(false);
        }

        /**
         * Enable checkbox to limit overlay to result image and no overlay
         */
        public void enableOverlayCheckbox(){
            overlayCheckbox.setEnabled(true);
        }

        /**
         * Check if overlay checkbox is selected.
         * @return true if overlay checkbox is selected
         */
        public boolean ovCheckbox(){
            return overlayCheckbox.isSelected();
        }

        /**
         * unchecks overlay checkbox
         */
        public void uncheckOvCheckbox(){
            overlayCheckbox.setSelected(false);
        }

        /**
         * Repaints all elements in GUI
         */
        public void repaintAll(){
            this.annotationsPanel.repaint();
            getCanvas().repaint();
            this.controlsPanel.repaint();
            this.all.repaint();
        }

        /**
         * Update the example lists in the GUI
         */
        protected void updateDisplayedLists()
        {
            for(int i = 0; i < numClasses; i++)
            {
                displayedLists[i].removeAll();
                for(int j=0;j<aRoiList[inputImage.getCurrentSlice()-1].get(i).size();++j){
                    displayedLists[i].add(new String("Trace "+j+" -z= "+inputImage.getCurrentSlice()));
                }
            }
        }
        /**
         * Overlay current traces on display image with 50% opacity.
         */
        void overlayTraces()
        {
        	boolean noTraces = true;
        	ColorProcessor cp = new ColorProcessor(inputImage.getWidth(),inputImage.getHeight());
            for(int i = 0; i < numClasses; i++)
            {
                for(int j=0;j<aRoiList[inputImage.getCurrentSlice()-1].get(i).size();++j){
                    Roi r = aRoiList[inputImage.getCurrentSlice()-1].get(i).get(j);
                    r.setStrokeColor(colors[i]);
                    r.setFillColor(colors[i]);
                    cp.drawRoi(r);
                    noTraces = false;
                }
            }
            if( noTraces )
            	return;
        	ImageRoi imgRoi = new ImageRoi(0,0,cp);
            imgRoi.setZeroTransparent(true);
            imgRoi.setOpacity( 0.5 );
            Overlay overlayList = inputImage.getOverlay();
            if( null == overlayList )
            	overlayList = new Overlay();
            overlayList.add(imgRoi);
            inputImage.setOverlay( overlayList );
        }
    }

    /**
     * Load training data from a file.
     */
    private void loadTrainingData() {
        win.disableAllButtons();
        Instances data=null;
        OpenDialog od = new OpenDialog("Choose data file", OpenDialog.getLastDirectory(), "data.arff");
        if (od.getFileName()==null) {
            win.setButtonsEnabled(0);
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
                win.setButtonsEnabled(0);
                return;
            }
        }
        catch(FileNotFoundException e){
            IJ.showMessage("File not found!");
            win.setButtonsEnabled(0);
            return;
        }
        try {
            if (null != data) {
                Attribute classAttribute = data.classAttribute();
                Enumeration<Object> classValues = classAttribute.enumerateValues();
                ArrayList<String> loadedClassNames = new ArrayList<String>();
                if (classAttribute.numValues() != numClasses) {
                    IJ.error("ERROR: Loaded number of classes and current number do not match!\n\tExpected number of classes: "+classAttribute.numValues()+"\n\tCurrent number of classes: "+numClasses);
                    trainingDataLoaded = false;
                    win.setButtonsEnabled(0);
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
                        win.setButtonsEnabled(0);
                        return;
                    }
                    j++;
                }
                //Select only loaded features
                Enumeration<Attribute> attributes = data.enumerateAttributes();
                ArrayList<RegionFeatures.Feature> loadedFeatures = new ArrayList<>();
                while(attributes.hasMoreElements())
                {
                    final Attribute a = attributes.nextElement();
                    String n = a.name();
                    if(a.name().endsWith("-l")||
                            a.name().endsWith("-a")||
                            a.name().endsWith("-b")){
                        n = a.name().substring(0,a.name().length()-2);
                    }
                    if(!loadedFeatures.contains(RegionFeatures.Feature.fromLabel(n))) {
                        loadedFeatures.add(RegionFeatures.Feature.fromLabel(n));
                    }
                }
                if(features.size()==loadedFeatures.size()){
                    for(int i=0;i<features.size();++i){
                        if(0!=features.get(i).toString().compareTo(loadedFeatures.get(i).toString())){
                            calculateFeatures=true; //Calculate features for new selected features
                            features=loadedFeatures;
                            break;
                        }
                    }
                }
                else
                {
                	calculateFeatures = true;
                	features = loadedFeatures;
                }

                if(loadedTrainingData==null) {
                    loadedTrainingData = data;
                }else {
                    IJ.log("Merging previously loaded data -"+loadedTrainingData.numInstances()+" instances- with new data -"+data.numInstances()+" instances-");
                    loadedTrainingData = eus.ehu.tss.Utils.merge(loadedTrainingData,data);
                }
                trainingDataLoaded = true;
                trainableSuperpixelSegmentation.setSelectedFeatures(features);
                IJ.log("Data loaded.");
                win.setButtonsEnabled(0);
            }

        }catch (Exception e){
            win.setButtonsEnabled(0);
            IJ.log("Error when loading training data!");
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
        trainableSuperpixelSegmentation.setClasses(classes);
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
        if(command=="Train classifier") {
            win.disableAllButtons();
            win.trainClassButton.setText("STOP");
            win.trainClassButton.setEnabled(true);
            Thread newTask = new Thread() {

                public void run() {
                    boolean regionSelected = false;
                    ArrayList<int[]> tags = new ArrayList<int[]>();
                    for(int i=0;i<numClasses;++i){
                        ArrayList<Integer> t = new ArrayList<>();
                        for(int l=0;l<inputImage.getNSlices();++l) {
                            for (int j = 0; j < aRoiList[l].get(i).size(); ++j) {
                                supImage.setSlice(l+1);
                                ArrayList<Float> floats = LabelImages.getSelectedLabels(supImage, aRoiList[l].get(i).get(j));
                                for (int k = 0; k < floats.size(); ++k) {
                                    t.add(floats.get(k).intValue());
                                    regionSelected = true;
                                }
                            }
                        }
                        int[] tg = new int[t.size()];
                        for (int j = 0; j < tg.length; ++j) {
                            tg[j] = t.get(j);
                        }
                        tags.add(tg);
                    }
                    if (!trainingDataLoaded) {

                        for (int i = 0; i < tags.size(); ++i) {
                            if (tags.get(i).length == 0) {
                                IJ.showMessage("Add at least one region to class " + classes.get(i));
                                win.trainClassButton.setText("Train classifier");
                                win.setButtonsEnabled(0);
                                return;
                            }
                        }
                    }
                    trainableSuperpixelSegmentation.setClasses(classes);
                    if (calculateFeatures) {
                        IJ.log("Calculating region features...");
                        trainableSuperpixelSegmentation.calculateRegionFeatures();
                        calculateFeatures = false;
                    }
                    try {
                        if (trainingDataLoaded) {
                            if(regionSelected) {
                                ArrayList<Attribute> attributes = new ArrayList<Attribute>();
                                ResultsTable unlabeledTable = trainableSuperpixelSegmentation.getUnlabeledTable();
                                int numFeatures = unlabeledTable.getLastColumn() + 1;
                                for (int i = 0; i < numFeatures; ++i) {
                                    attributes.add(new Attribute(unlabeledTable.getColumnHeading(i), i));
                                }
                                attributes.add(new Attribute("Class", classes));
                                Instances trainingData = new Instances("training data", attributes, 0);
                                int[] labels = LabelImages.findAllLabels(supImage);
                                HashMap<Integer, Integer> labelIndices = LabelImages.mapLabelIndices(labels);

                                for (int i = 0; i < tags.size(); ++i) { //For each class in classRegions
                                    for (int j = 0; j < tags.get(i).length; ++j) {
                                        Instance inst = new DenseInstance(numFeatures + 1);
                                        for (int k = 0; k < numFeatures; ++k) {
                                            inst.setValue(k, unlabeledTable.getValueAsDouble(k,
                                                    labelIndices.get(
                                                            tags.get(i)[j]
                                                    )));
                                        }
                                        inst.setValue(numFeatures, i); // set class value
                                        trainingData.add(inst);
                                    }
                                }
                                trainingData.setClassIndex(numFeatures); // set class index
                                IJ.log("Merging previously loaded data -" + loadedTrainingData.numInstances() + " instances- with selected regions -" + trainingData.numInstances() + " instances-");
                                trainingData = eus.ehu.tss.Utils.merge(trainingData, loadedTrainingData);
                                trainableSuperpixelSegmentation.setTrainingData(trainingData);
                                IJ.log("Training classifier with " + trainableSuperpixelSegmentation.getTrainingData().numInstances() + " instances...");
                                if (!trainableSuperpixelSegmentation.trainClassifier()) {
                                    IJ.error("Error when training classifier");
                                }
                            }else {
                                trainableSuperpixelSegmentation.setTrainingData(loadedTrainingData);
                                IJ.log("Training classifier with " + trainableSuperpixelSegmentation.getTrainingData().numInstances() + " instances...");
                                if (!trainableSuperpixelSegmentation.trainClassifier()) {
                                    IJ.error("Error when training classifier");
                                }
                            }
                        }else {
                            trainableSuperpixelSegmentation.setTrainingData(null);
                            trainableSuperpixelSegmentation.calculateTrainingData(tags);
                            IJ.log("Training classifier with "+trainableSuperpixelSegmentation.getTrainingData().numInstances()+" instances...");
                            trainableSuperpixelSegmentation.trainClassifier();
                        }
                    } catch (Exception e) {
                        e.printStackTrace();
                        IJ.log("Error when merging loaded training data selected data!");
                        win.trainClassButton.setText("Train classifier");
                        win.setButtonsEnabled(0);
                        return;
                    }
                    classifier = trainableSuperpixelSegmentation.getClassifier();
                    IJ.log("Classifier trained.");
                    IJ.log("Applying classifier...");
                    resultImage = trainableSuperpixelSegmentation.applyClassifier();
                    currentOverlay = OverlayMode.SUPERPIXEL_IMAGE;
                    toggleOverlay();
                    IJ.log("Classifier applied.");
                    win.enableOverlayCheckbox();
                    win.setButtonsEnabled(2);
                    win.trainClassButton.setText("Train classifier");
                }
            };
            trainingTask = newTask;
            newTask.start();
        }else if(command.equals("STOP")){
            try{

                IJ.log("Training was stopped by the user!");
                win.setButtonsEnabled(0);
                win.trainClassButton.setText("Train classifier");

                if(null != trainingTask) {
                    trainingTask.interrupt();
                    trainableSuperpixelSegmentation = new TrainableSuperpixelSegmentation(inputImage, supImage, features, classifier, classes);
                    trainableSuperpixelSegmentation.setClassifierTrained(false);
                    calculateFeatures = true;
                }else {
                    IJ.log("Error: interrupting training failed becaused the thread is null!");
                    win.trainClassButton.setText("Train classifier");
                    win.setButtonsEnabled(0);
                    return;
                }

            }catch(Exception ex){
                ex.printStackTrace();
                win.trainClassButton.setText("Train classifier");
                win.setButtonsEnabled(0);
                return;
            }
        }
    }


    /**
     * Apply classifier to loaded image and corresponding superpixel image
     */
    void applyClassifier(){
        win.disableAllButtons();
        if(!trainableSuperpixelSegmentation.isClassifierTrained()){
            IJ.error("Train a classifier");
            win.setButtonsEnabled(0);
            return;
        }
        IJ.showMessage("Select input image");
        ImagePlus input = IJ.openImage();
        if(input==null){
            win.setButtonsEnabled(0);
            IJ.log("Classifier applying was cancelled.");
            return;
        }
        IJ.showMessage("Select corresponding label image");
        ImagePlus sup = IJ.openImage();
        if(sup==null){
            win.setButtonsEnabled(0);
            IJ.log("Classifier applying was cancelled.");
            return;
        }
        trainableSuperpixelSegmentation.setInputImage(input);
        trainableSuperpixelSegmentation.setLabelImage(sup);
        IJ.log("Calculating region features...");
        trainableSuperpixelSegmentation.calculateRegionFeatures();
        IJ.log("Applying classifier...");
        ImagePlus resultImg = trainableSuperpixelSegmentation.applyClassifier();
        eus.ehu.tss.Utils.convertTo8bitNoScaling(resultImg);
        resultImg.getProcessor().setColorModel(overlayLUT);
        resultImg.getImageStack().setColorModel(overlayLUT);
        resultImg.updateAndDraw();
        resultImg.show();
        IJ.log("Classifier applied.");
        win.setButtonsEnabled(1);
    }


    /**
     * Save instances into ARFF file
     */
    void saveInstances(){
        win.disableAllButtons();
        try {
            if(calculateFeatures){
                IJ.log("Calculating region features...");
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
            SaveDialog sd = new SaveDialog("Save instances as...",inputTitle+"-trainingData",".arff");
            if(sd.getFileName()==null){
                win.setButtonsEnabled(0);
                return;
            }
            String filename = sd.getDirectory() + sd.getFileName();
            saver.setFile(new File(filename));
            saver.writeBatch();
            IJ.log("File saved at "+sd.getDirectory()+sd.getFileName());
            win.setButtonsEnabled(0);
        } catch (IOException e) {
            win.setButtonsEnabled(0);
            e.printStackTrace();
        }

    }

    /**
     * Save classifier to file
     */
    void saveClassifier(){
        win.disableAllButtons();

        if(!trainableSuperpixelSegmentation.isClassifierTrained()){
            IJ.error("Train classifier");
            win.setButtonsEnabled(0);
            return;
        }
        SaveDialog sd = new SaveDialog("Save model as...",inputTitle+"-classifier",".model");
        if(sd.getFileName()==null){
            win.setButtonsEnabled(0);
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
            win.setButtonsEnabled(0);
            IJ.error("Save Failed", "Error when saving classifier into a file");
            saveOK = false;
        }
        if (saveOK)
            win.setButtonsEnabled(0);
        IJ.log("Saved model into " + filename );


    }

    /**
     * Load classifier from file
     */
    void loadClassifier(){
        win.disableAllButtons();
        OpenDialog od = new OpenDialog( "Choose Weka classifier file", "" );
        if (od.getFileName()==null) {
            win.setButtonsEnabled(0);
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
                ArrayList<String> loadedClassNames = new ArrayList<String>();
                if (classAttribute.numValues() != numClasses) {
                    IJ.error("ERROR: Loaded number of classes and current number do not match!\n\tExpected number of classes: "+classAttribute.numValues()+"\n\tCurrent number of classes: "+numClasses);
                    win.setButtonsEnabled(0);
                    return;
                }
                int j = 0;
                while (classValues.hasMoreElements()) {
                    final String className = ((String) classValues.nextElement()).trim();
                    loadedClassNames.add(className);

                    IJ.log("Read class name: " + className);
                    if (!className.equals(classes.get(j))) {
                        IJ.error("ERROR: Loaded classes and current classes do not match!\n\tExpected: " + className + "\n\tFound: " + classes.get(j));
                        win.setButtonsEnabled(0);
                        return;
                    }
                    j++;
                }
                trainableSuperpixelSegmentation.setClassifier(loadedClassifier);
            }catch (Exception e){
            }
            objectInputStream.close();
        }catch (Exception e){
            IJ.log("Loading file failed: "+e.getMessage());
            win.setButtonsEnabled(0);
            return;
        }
        classifier = loadedClassifier;
        trainableSuperpixelSegmentation.setClasses(classes);
        calculateFeatures = true;
        resultImage=null;
        currentOverlay = OverlayMode.RESULT_IMAGE;
        if(win.ovCheckbox()){
            win.uncheckOvCheckbox();
        }
        toggleOverlay();
        IJ.log(loadedClassifier.toString());
        trainableSuperpixelSegmentation.setClassifierTrained(true);
        win.setButtonsEnabled(1);

    }

    /**
     *Create result image, trains classifier if it hasn't been trained before
     */
    void createResult(){
        win.disableAllButtons();
        try {
            if (resultImage == null) {
                if (calculateFeatures) {
                    trainableSuperpixelSegmentation.calculateRegionFeatures();
                    calculateFeatures = false;
                }
                if (!trainableSuperpixelSegmentation.isClassifierTrained()) {
                    runStopTraining("Train classifier");
                } else {
                    resultImage = trainableSuperpixelSegmentation.applyClassifier();
                    createResult();
                }
            } else {

                ImagePlus resultImg = resultImage.duplicate();


                resultImg.setTitle(inputTitle + "-classified");

                eus.ehu.tss.Utils.convertTo8bitNoScaling(resultImg);
                resultImg.getProcessor().setColorModel(overlayLUT);
                resultImg.getImageStack().setColorModel(overlayLUT);
                resultImg.updateAndDraw();
                resultImg.show();
                win.enableOverlayCheckbox();
            }
            win.setButtonsEnabled(2);
        }catch (Exception e){
            e.printStackTrace();
            IJ.log("Error when creating result!");
            win.setButtonsEnabled(0);
        }
    }


    /**
     * Shows dialog with feature, classifier and overlay opacity options
     */
    void showSettingsDialog(){
        win.disableAllButtons();
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
            IJ.log("Feature selection disabled because training data has been loaded.");
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
            gd.addStringField("Class "+i, classes.get(i), 15);

        JButton createFeatureImage = new JButton("Create feature image");
        createFeatureImage.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if(calculateFeatures){
                    IJ.log("Calculating region features...");
                    trainableSuperpixelSegmentation.calculateRegionFeatures();
                    calculateFeatures = false;
                }
                ImagePlus featureImage = trainableSuperpixelSegmentation.getFeatureImage(supImage,trainableSuperpixelSegmentation.getUnlabeledTable());
                featureImage.show();
            }
        });
        gd.add(createFeatureImage);

        gd.showDialog();

        if(gd.wasCanceled()){
            win.setButtonsEnabled(0);
            return;
        }
        int numSelectedFeatures =0;
        ArrayList<RegionFeatures.Feature> newFeatures = new ArrayList<RegionFeatures.Feature>();
        for(int i=0;i<RegionFeatures.totalFeatures();++i){
            enabledFeatures[i] = gd.getNextBoolean();
            if(enabledFeatures[i]){
                newFeatures.add(RegionFeatures.Feature.fromLabel(avFeatures[i]));
                numSelectedFeatures++;
            }
        }
        if(numSelectedFeatures==0){
            IJ.error("Select at least one training feature");
            IJ.log("Default features selected.");
            for(int i=0;i<RegionFeatures.totalFeatures();++i){
                newFeatures.add(RegionFeatures.Feature.fromLabel(avFeatures[i]));
                numSelectedFeatures++;
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
                win.setButtonsEnabled(0);
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
            win.updateOverlay();
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
        win.setButtonsEnabled(0);


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
            IJ.error( "Error in plot result",
                    "No data available yet to plot results: you need to train a classifier" );
            IJ.log("Failed to display plot!");
            return;
        }
        displayGraphs(data,trainableSuperpixelSegmentation.getClassifier());
        IJ.showStatus("Done.");
        IJ.log("Done.");
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
        try {
            win.disableAllButtons();
            ArrayList<int[]> tags = new ArrayList<int[]>();
            for(int i=0;i<numClasses;++i){
                ArrayList<Integer> t = new ArrayList<>();
                for(int l=0;l<inputImage.getNSlices();++l) {
                    for (int j = 0; j < aRoiList[l].get(i).size(); ++j) {
                        supImage.setSlice(l+1);
                        ArrayList<Float> floats = LabelImages.getSelectedLabels(supImage, aRoiList[l].get(i).get(j));
                        for (int k = 0; k < floats.size(); ++k) {
                            t.add(floats.get(k).intValue());
                        }
                    }
                    int[] tg = new int[t.size()];
                    for (int j = 0; j < tg.length; ++j) {
                        tg[j] = t.get(j);
                    }
                    tags.add(tg);
                }

            }
            trainableSuperpixelSegmentation.setClasses(classes);
            if (calculateFeatures) {
                IJ.log("Calculating region features...");
                trainableSuperpixelSegmentation.calculateRegionFeatures();
                calculateFeatures = false;
            }
            if (!trainableSuperpixelSegmentation.isClassifierTrained()) {
                if (!trainingDataLoaded) {
                    for (int i = 0; i < tags.size(); ++i) {
                        if (tags.get(i).length == 0) {
                            IJ.showMessage("Add at least one region to class " + classes.get(i));
                            win.setButtonsEnabled(0);
                            return;
                        }
                    }
                }
                Instances unlabeled = trainableSuperpixelSegmentation.getUnlabeled();
                ArrayList<Attribute> attributes = new ArrayList<Attribute>();
                int numFeatures = unlabeled.numAttributes() - 1;
                for (int i = 0; i < numFeatures; ++i) {
                    attributes.add(new Attribute(unlabeled.attribute(i).name(), i));
                }
                attributes.add(new Attribute("Class", classes));
                Instances trainingData = new Instances("training data", attributes, 0);
                // Fill training dataset with the feature vectors of the corresponding
                // regions given by classRegions
                for (int i = 0; i < tags.size(); ++i) { //For each class in classRegions
                    for (int j = 0; j < tags.get(i).length; ++j) {
                        Instance inst = new DenseInstance(numFeatures + 1);
                        for (int k = 0; k < numFeatures; ++k) {
                            inst.setValue(k, unlabeled.get(tags.get(i)[j] - 1).value(k));
                        }
                        inst.setValue(numFeatures, i); // set class value
                        trainingData.add(inst);
                    }
                }
                trainingData.setClassIndex(numFeatures); // set class index
                try {
                    if (trainingDataLoaded) {
                        IJ.log("Merging previously loaded data -" + loadedTrainingData.numInstances() + " instances- with selected regions -" + trainingData.numInstances() + " instances-");
                        trainingData = eus.ehu.tss.Utils.merge(trainingData, loadedTrainingData);
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                    IJ.log("Error when merging loaded training data selected data!");
                }
                trainableSuperpixelSegmentation.setTrainingData(trainingData);
                IJ.log("Training classifier with " + trainableSuperpixelSegmentation.getTrainingData().numInstances() + " instances...");
                if (!trainableSuperpixelSegmentation.trainClassifier()) {
                    IJ.error("Error when training classifier");
                    win.setButtonsEnabled(0);
                    return;
                }
                classifier = trainableSuperpixelSegmentation.getClassifier();
            }
            ImagePlus probabilityImage = trainableSuperpixelSegmentation.getProbabilityMap();
            probabilityImage.setTitle(inputTitle + "-prob");
            probabilityImage.show();
            win.setButtonsEnabled(0);
        }catch (Exception e){
            e.printStackTrace();
            return;
        }

    }

    /**
     * Changes displayed overlay, between no overlay, superpixel overlay and result overlay
     */
    void toggleOverlay()
    {
    	// Take action based on current overlay
    	switch( currentOverlay )
    	{
	    	case INPUT_IMAGE:
	    		if( resultImage == null )
	    		{
                	currentOverlay = OverlayMode.SUPERPIXEL_IMAGE;
                }
	    		else
	    		{
                	if( win.ovCheckbox() )
    	    			currentOverlay = OverlayMode.RESULT_IMAGE;
    	            else
    	            	currentOverlay = OverlayMode.SUPERPIXEL_IMAGE;
                }
	    		break;
	    	case RESULT_IMAGE:
	    		currentOverlay = OverlayMode.INPUT_IMAGE;
	    		break;
	    	case SUPERPIXEL_IMAGE:
	    		if( resultImage != null )
	    			currentOverlay = OverlayMode.RESULT_IMAGE;
	    		else
	    			currentOverlay = OverlayMode.INPUT_IMAGE;
	    		break;
	    	default:
	    		break;
    	}
    	// Display new overlay
    	win.updateOverlay();
    }
    /**
     * Delete selected tag
     * @param i identifier of class to remove tags
     */
    void deleteSelected(final int i){
        try {
            int index = displayedLists[i].getSelectedIndex();
            aRoiList[inputImage.getCurrentSlice()-1].get(i).remove(index);
            win.updateDisplayedLists();
            win.updateOverlay();
            inputImage.killRoi();
        }catch (Exception e1){
            e1.printStackTrace();
        }
    }
    /**
     * Adds tags based on the ROIs selected by the user
     * @param i identifier of class to add tags
     */
    void addExamples(int i){
        try {
            final Roi r = inputImage.getRoi();
            r.setStrokeColor(colors[i]);
            aRoiList[inputImage.getCurrentSlice()-1].get(i).add(r);
            win.updateDisplayedLists();
            win.updateOverlay();
            inputImage.killRoi();
        }catch (Exception e){
            e.printStackTrace();
        }
    }

    /**
     * Displays selected Roi
     * @param e event associated to the list
     * @param i index of the selected list
     */
    void listSelected(final ItemEvent e, final int i)
    {
    	// Index of the selected trace in the specific list
        int selectedIndex = displayedLists[i].getSelectedIndex();
        // Selected ROI
        final Roi newRoi = aRoiList[inputImage.getCurrentSlice()-1].get(i).get(selectedIndex);
        // Display ROI in image
        newRoi.setImage(inputImage);
        inputImage.setRoi(newRoi);
        inputImage.updateAndDraw();
        // Deselect rest of list and traces
        for(int j=0;j<displayedLists[i].getItemCount();++j) {
            if(j!=selectedIndex) {
                displayedLists[i].deselect(j);
            }
        }
        for(int c=0;c<numClasses;++c){
            if(c!=i){
               for(int j=0;j<displayedLists[c].getItemCount();++j) {
                   displayedLists[c].deselect(j);
               }
            }
        }
    }


    @Override
    public void run(String s) {

        classes = new ArrayList<String>();
        // Check number of open images
        int nbima = WindowManager.getImageCount();
        // If less than 2 images opened, ask user
        if( nbima < 2 )
        {
            IJ.log("Opening input image...");
            inputImage = IJ.openImage();
            IJ.log("Opening superpixel image...");
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
            }
            else
                return;
        }
        if( inputImage.getWidth() != supImage.getWidth() ||
                inputImage.getHeight() != supImage.getHeight() ||
                inputImage.getImageStackSize() != supImage.getImageStackSize() )
        {
            IJ.error( "Trainable Superpixel Segmentation input error", "Error: input"
                    + " and superpixel images must have the same size" );
            return;
        }
        inputTitle = inputImage.getShortTitle();
        inputImage.setTitle("Trainable Superpixel Segmentation");

        if(inputImage == null || supImage == null){
            IJ.error("Error when opening image");
        }else {

            aRoiList = new ArrayList[inputImage.getNSlices()];
            for(int j=0;j<inputImage.getNSlices();++j) {
                aRoiList[j] = new ArrayList<ArrayList<Roi>>();
            }
            for (int i = 0; i < numClasses; ++i) {
                displayedLists[i] = new java.awt.List(5);
                displayedLists[i].setForeground(colors[i]);
            }

            for (int i = 0; i < numClasses; ++i) {
                classes.add("class " + i);
                for (int j = 0; j < inputImage.getNSlices(); ++j) {
                    aRoiList[j].add(new ArrayList<>());
                }
            }

            features = new ArrayList<>();
            String[] selectedFs = RegionFeatures.Feature.getAllLabels();
            for(int i=0;i<selectedFs.length;++i){
                features.add(RegionFeatures.Feature.fromLabel(selectedFs[i]));
            }
            // Define classifier
            classifier = new RandomForest();
            trainableSuperpixelSegmentation = new TrainableSuperpixelSegmentation(inputImage,supImage,features,classifier,classes);
            supImage = trainableSuperpixelSegmentation.getLabelImage(); //Update after remaping
            // Build GUI
    		SwingUtilities.invokeLater(
    				new Runnable() {
    					public void run() {
    						win = new CustomWindow( inputImage );
    						win.pack();
    						win.setButtonsEnabled(0);
    					}
    				});
            Toolbar.getInstance().setTool( Toolbar.FREELINE );
        }
    }
    /**
     * Main static method.
     * @param args main arguments
     */
    public static void main(String[] args){
        Class<?> clazz = Trainable_Superpixel_Segmentation.class;
        String url = clazz.getResource("/" + clazz.getName().replace('.', '/') + ".class").toString();
        String pluginsDir = url.substring("file:".length(), url.length() - clazz.getName().length() - ".class".length());
        System.setProperty("plugins.dir", pluginsDir);
        IJ.runPlugIn(clazz.getName(),"");

    }


}