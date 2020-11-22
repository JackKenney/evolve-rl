package model

// Model contains the learning model and provides the interface to interact with it.
type Model interface {
	Predict(data interface{}) (interface{}, error)
	Train(data interface{}) (interface{}, error)
}

// MockModel provides the methods that are required of a Model
// to demonstrate how to inject the the Model into the handler function.
type MockModel struct{}

// ProvideMockModel returns a mocked Model interface.
func ProvideMockModel() *MockModel {
	return &MockModel{}
}

// Predict returns mock prediction for input data.
func (m *MockModel) Predict(data interface{}) (interface{}, error) {
	return nil, nil
}

// Train returns an interface with the data expected from this model
// and any errors that occurred during training
func (m *MockModel) Train(data interface{}) (interface{}, error) {
	return nil, nil
} 