#pragma once

class LWO3NodeGraph
{
 public:
    // Initialize with root form and specific surface form
    LWO3NodeGraph (std::shared_ptr<LWO3Form> root, const LWO3Form* surfForm = nullptr);

   
    // Get all nodes grouped by type
    std::map<std::string, std::vector<std::string>> getNodesByType() const { return nodesByType_; }

    // Get info about all nodes including their connections
    std::vector<ImageNodeInfo> getImageNodes() const;
    std::vector<StandardNodeInfo> getStandardNodes() const;
    std::vector<PrincipledBSDFInfo> getPrincipledBSDFNodes() const;

    // Get connections for a specific node
    std::vector<NodeConnection> getNodeConnections (const std::string& nodeName) const;

    // Get all connected image nodes with their BSDF inputs
    std::vector<ImageConnection> getConnectedImageNodes() const;

    // Utility functions
    size_t getNodeCount() const { return nodeNames_.size(); }
    bool hasNode (const std::string& nodeName) const;
    const std::string& getSurfaceName() const { return surfaceName_; }
    const LWO3Form* getRootForm() const { return root_.get(); }

 private:
    std::shared_ptr<LWO3Form> root_;
    const LWO3Form* surfForm_; // The specific surface form we're working with
    std::string surfaceName_;
    std::vector<std::pair<std::string, size_t>> nodeNames_;
    std::map<std::string, std::vector<std::string>> nodesByType_;

    void extractSurfaceName();
    void extractNodeNames();

    // Helper functions for node data extraction
    std::string extractImagePathFromNDTA (const LWO3Form* ndtaForm) const;
    std::vector<NodeConnection> processNodeConnections (const LWO3Form* nconForm,
                                                        const std::string& nodeName) const;
    
    void extractStandardValues (const LWO3Form* ndtaForm,
                                StandardNodeInfo& info) const;

    void extractPrincipledBSDFValues (const LWO3Form* ndtaForm,
                                      PrincipledBSDFInfo& info) const;
};

